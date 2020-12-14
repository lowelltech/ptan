import ptan
import enum
import numpy as np
import time
from typing import Optional
from ignite.engine import Engine, State
from ignite.engine import Events, EventEnum
from ignite.handlers.timing import Timer


class EpisodeEvents(EventEnum):
    EPISODE_COMPLETED = "episode_completed"
    BOUND_REWARD_REACHED = "bound_reward_reached"
    BEST_REWARD_REACHED = "best_reward_reached"


class EndOfEpisodeHandler:
    def __init__(self, exp_source: ptan.experience.ExperienceSource, alpha: float = 0.98,
                 bound_avg_reward: Optional[float] = None,
                 bound_mean_reward: Optional[float] = None,
                 stat_size: int = 100,
                 subsample_end_of_episode: Optional[int] = None):
        """
        Construct end-of-episode event handler
        :param exp_source: experience source to use
        :param alpha: smoothing alpha param
        :param bound_avg_reward: optional boundary for average reward
        :param bound_mean_reward: optional boundary for mean reward
        :param stat_size: number of values for mean, std statistics; <= 1 to disable
        :param subsample_end_of_episode: if given, end of episode event will be subsampled by this amount
        """
        self._exp_source = exp_source
        self._alpha = alpha
        self._bound_avg_reward = bound_avg_reward
        self._bound_mean_reward = bound_mean_reward
        self._best_avg_reward = None
        self._subsample_end_of_episode = subsample_end_of_episode
        self._stat_size = stat_size
        self._last_idx = 0
        if self._stat_size > 1:
            self._last_rewards = np.empty(self._stat_size, dtype=np.float32)
            self._last_rewards[:] = np.NaN
            self._last_steps = np.empty(self._stat_size, dtype=np.float32)
            self._last_steps[:] = np.NaN

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        engine.register_events(*EpisodeEvents)
        State.event_to_attr[EpisodeEvents.EPISODE_COMPLETED] = "episode"
        State.event_to_attr[EpisodeEvents.BOUND_REWARD_REACHED] = "episode"
        State.event_to_attr[EpisodeEvents.BEST_REWARD_REACHED] = "episode"

    def __call__(self, engine: Engine):
        for reward, steps in self._exp_source.pop_rewards_steps():
            engine.state.episode = getattr(engine.state, "episode", 0) + 1
            engine.state.episode_reward = reward
            engine.state.episode_steps = steps
            engine.state.metrics['reward'] = reward
            engine.state.metrics['steps'] = steps

            if self._stat_size > 1:
                # Keep stat_size elements in numpy arrays.
                if (self._last_idx < 0) or (self._last_idx >= self._stat_size):
                    self._last_idx = 0
                self._last_rewards[self._last_idx] = reward
                self._last_steps[self._last_idx] = steps
                self._last_idx += 1

                # Compute mean and standard deviation (nan* ignores NaN elements)
                engine.state.metrics['mean_reward'] = np.nanmean(self._last_rewards)
                engine.state.metrics['mean_steps'] = np.nanmean(self._last_steps)
                engine.state.metrics['std_reward'] = np.nanstd(self._last_rewards)
                engine.state.metrics['std_steps'] = np.nanstd(self._last_steps)

            self._update_smoothed_metrics(engine, reward, steps)
            if self._subsample_end_of_episode is None or engine.state.episode % self._subsample_end_of_episode == 0:
                engine.fire_event(EpisodeEvents.EPISODE_COMPLETED)
            if self._bound_avg_reward is not None and engine.state.metrics['avg_reward'] >= self._bound_avg_reward:
                engine.fire_event(EpisodeEvents.BOUND_REWARD_REACHED)
            if self._bound_mean_reward is not None and self._stat_size >= 1 and engine.state.metrics['mean_reward'] >= self._bound_mean_reward:
                engine.fire_event(EpisodeEvents.BOUND_REWARD_REACHED)
            if self._best_avg_reward is None:
                self._best_avg_reward = engine.state.metrics['avg_reward']
            elif self._best_avg_reward < engine.state.metrics['avg_reward']:
                engine.fire_event(EpisodeEvents.BEST_REWARD_REACHED)
                self._best_avg_reward = engine.state.metrics['avg_reward']

    def _update_smoothed_metrics(self, engine: Engine, reward: float, steps: int):
        for attr_name, val in zip(('avg_reward', 'avg_steps'), (reward, steps)):
            if attr_name not in engine.state.metrics:
                engine.state.metrics[attr_name] = val
            else:
                engine.state.metrics[attr_name] *= self._alpha
                engine.state.metrics[attr_name] += (1-self._alpha) * val


class EpisodeFPSHandler:
    FPS_METRIC = 'fps'
    AVG_FPS_METRIC = 'avg_fps'
    TIME_PASSED_METRIC = 'time_passed'

    def __init__(self, fps_mul: float = 1.0, fps_smooth_alpha: float = 0.98):
        self._timer = Timer(average=True)
        self._fps_mul = fps_mul
        self._started_ts = time.time()
        self._fps_smooth_alpha = fps_smooth_alpha

    def attach(self, engine: Engine, manual_step: bool = False):
        self._timer.attach(engine, step=None if manual_step else Events.ITERATION_COMPLETED)
        engine.add_event_handler(EpisodeEvents.EPISODE_COMPLETED, self)

    def step(self):
        """
        If manual_step=True on attach(), this method should be used every time we've communicated with environment
        to get proper FPS
        :return:
        """
        self._timer.step()

    def __call__(self, engine: Engine):
        t_val = self._timer.value()
        if engine.state.iteration > 1:
            fps = self._fps_mul / t_val
            avg_fps = engine.state.metrics.get(self.AVG_FPS_METRIC)
            if avg_fps is None:
                avg_fps = fps
            else:
                avg_fps *= self._fps_smooth_alpha
                avg_fps += (1-self._fps_smooth_alpha) * fps
            engine.state.metrics[self.AVG_FPS_METRIC] = avg_fps
            engine.state.metrics[self.FPS_METRIC] = fps
        engine.state.metrics[self.TIME_PASSED_METRIC] = time.time() - self._started_ts
        self._timer.reset()


class PeriodEvents(EventEnum):
    ITERS_10_COMPLETED = "iterations_10_completed"
    ITERS_100_COMPLETED = "iterations_100_completed"
    ITERS_1000_COMPLETED = "iterations_1000_completed"
    ITERS_10000_COMPLETED = "iterations_10000_completed"
    ITERS_100000_COMPLETED = "iterations_100000_completed"


class PeriodicEvents:
    """
    The same as CustomPeriodicEvent from ignite.contrib, but use true amount of iterations,
    which is good for TensorBoard
    """

    INTERVAL_TO_EVENT = {
        10: PeriodEvents.ITERS_10_COMPLETED,
        100: PeriodEvents.ITERS_100_COMPLETED,
        1000: PeriodEvents.ITERS_1000_COMPLETED,
        10000: PeriodEvents.ITERS_10000_COMPLETED,
        100000: PeriodEvents.ITERS_100000_COMPLETED,
    }

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        engine.register_events(*PeriodEvents)
        for e in PeriodEvents:
            State.event_to_attr[e] = "iteration"

    def __call__(self, engine: Engine):
        for period, event in self.INTERVAL_TO_EVENT.items():
            if engine.state.iteration % period == 0:
                engine.fire_event(event)


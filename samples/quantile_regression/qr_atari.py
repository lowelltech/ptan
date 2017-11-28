#!/usr/bin/env python3
import ptan
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from lib import common, atari_wrappers

PLAY_STEPS = 4
# quantilles count
QUANT_N = 100
HUBER_K = 1


def make_env(params):
    env = atari_wrappers.make_atari(params['env_name'])
    env = atari_wrappers.wrap_deepmind(env, frame_stack=True, pytorch_img=True)
    return env


def play_func(params, net, cuda, exp_queue):
    env = make_env(params)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-qr")
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(lambda x: net.qvals(x), selector, cuda=cuda)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    exp_source_iter = iter(exp_source)

    frame_idx = 0

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            exp = next(exp_source_iter)
            exp_queue.put(exp)

            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

    exp_queue.put(None)


class QRDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QRDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions*QUANT_N)
        )

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1)
        return self.fc(conv_out).view(batch_size, -1, QUANT_N)

    def qvals(self, x):
        return self.qvals_from_quant(self(x))

    @classmethod
    def qvals_from_quant(cls, quant):
        return quant.mean(dim=2)


def calc_loss_qr(batch, net, tgt_net, gamma, cuda=False):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    done_indices = np.where(dones)[0]
    batch_size = len(batch)

    states_v = Variable(torch.from_numpy(states))
    next_states_v = Variable(torch.from_numpy(next_states), volatile=True)
    actions_v = Variable(torch.from_numpy(actions))
    rewards_v = Variable(torch.from_numpy(rewards))
    tau_hat_v = Variable(torch.range(0.0, 1.0 - 1/QUANT_N, 1/QUANT_N)) + 0.5/QUANT_N
    if cuda:
        states_v = states_v.cuda(async=True)
        next_states_v = next_states_v.cuda(async=True)
        actions_v = actions_v.cuda(async=True)
        rewards_v = rewards_v.cuda(async=True)
        tau_hat_v = tau_hat_v.cuda(async=True)

    next_quant_v = tgt_net(next_states_v)
    best_actions_v = tgt_net.qvals_from_quant(next_quant_v).max(1)[1]
    best_next_quant_v = next_quant_v[range(batch_size), best_actions_v.data]
    if dones.any():
        done_indices_v = torch.from_numpy(done_indices)
        if cuda:
            done_indices_v = done_indices_v.cuda()
        best_next_quant_v[done_indices_v] = 0.0
    best_next_quant_v.volatile = False
    expected_quant_v = best_next_quant_v * gamma + rewards_v.unsqueeze(-1)
    quant_v = net(states_v)[range(batch_size), actions_v.data]
    u = expected_quant_v - quant_v

    abs_u = u.abs()
    mask_small_u = (abs_u <= HUBER_K).float()
    huber_loss = mask_small_u * 0.5 * (u ** 2)
    huber_loss = huber_loss + (1 - mask_small_u) * HUBER_K * (abs_u - HUBER_K / 2)

    huber_mul = torch.abs(tau_hat_v.unsqueeze(0) - (u < 0).float())
    final_loss = huber_mul * huber_loss
    return final_loss.sum() / QUANT_N


if __name__ == "__main__":
    mp.set_start_method('spawn')
    params = common.HYPERPARAMS['pong']
    params['batch_size'] *= PLAY_STEPS
    # params['batch_size'] = 8  # For debugging
    # params['replay_initial'] = 100
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()

    env = make_env(params)
    net = QRDQN(env.observation_space.shape, env.action_space.n)
    print(net)
    if args.cuda:
        net.cuda()

    tgt_net = ptan.agent.TargetNet(net)

    buffer = ptan.experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    exp_queue = mp.Queue(maxsize=PLAY_STEPS * 2)
    play_proc = mp.Process(target=play_func, args=(params, net, args.cuda, exp_queue))
    play_proc.start()

    frame_idx = 0
    batch_with_dones = None

    while play_proc.is_alive():
        frame_idx += PLAY_STEPS
        for _ in range(PLAY_STEPS):
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            buffer._add(exp)

        if len(buffer) < params['replay_initial']:
            continue

        optimizer.zero_grad()
        batch = buffer.sample(params['batch_size'])
        if batch_with_dones is None:
            if any(map(lambda t: t.last_state is None, batch)):
                print("Batch with dones found!")
                batch_with_dones = batch

        loss_v = calc_loss_qr(batch, net, tgt_net.target_model, gamma=params['gamma'], cuda=args.cuda)
        if loss_v is not None:
            loss_v.backward()
            optimizer.step()

        if frame_idx % params['target_net_sync'] < PLAY_STEPS:
            tgt_net.sync()

# execute: python -m scripts.plan_maze2d   --dataset maze2d-umaze-v1   --logbase /home/km6748/DiffuserV2/pretrained/logs   --diffusion_loadpath diffusion/H128_T64   --diffusion_epoch 1760000   --batch_size 128   --vis_freq 10   --conditional True

import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

#---------------------------------- main loop ----------------------------------#

USE_MODEL_ACTION = False

observation = env.reset()

if args.conditional:
    print('Resetting target')
    env.set_target()

target = env._target
cond = { diffusion.horizon - 1: np.array([*target, 0, 0]) }

rollout = [observation.copy()]
total_reward = 0

for t in range(env.max_episode_steps):
    state = env.state_vector().copy()

    if t == 0:
        cond[0] = observation                      # add the start state
        _, samples = policy(cond, batch_size=args.batch_size)
        actions   = samples.actions[0]             # shape: [H, action_dim]
        sequence  = samples.observations[0]

    if USE_MODEL_ACTION:
        if t < len(actions):
            action = actions[t]                        # open-loop execution
        else:
            action = np.zeros(env.action_space.shape)  # horizon exhausted
    else:
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

    next_observation, reward, terminal, _ = env.step(action)
    total_reward += reward
    score = env.get_normalized_score(total_reward)

    print(f't: {t} | r: {reward:.2f} | R: {total_reward:.2f} | '
          f'score: {score:.4f} | {action}')

    if 'maze2d' in args.dataset:
        xy   = next_observation[:2]
        goal = env.unwrapped._target
        print(f'maze | pos: {xy} | goal: {goal}')

    rollout.append(next_observation.copy())

    if t % args.vis_freq == 0 or terminal:
        fullpath = join(args.savepath, f'{t}.png')
        if t == 0:
            renderer.composite(fullpath, samples.observations, ncol=1)
        renderer.composite(join(args.savepath, 'rollout.png'),
                           np.array(rollout)[None], ncol=1)

    if terminal:
        break

    observation = next_observation


json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

import json
import numpy as np
from os.path import join
import pdb
import gym
from gym.vector import VectorEnv
from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'

# custom VectorEnv wrapper for Maze2D
class MazeVectorEnv(VectorEnv):
    def __init__(self, num_envs, dataset):
        self.num_envs = num_envs
        self.envs = [datasets.load_environment(dataset) for _ in range(num_envs)]
        obs_space = self.envs[0].observation_space
        act_space = self.envs[0].action_space
        super().__init__(num_envs, obs_space, act_space)
    def reset(self):
        states = [e.reset() for e in self.envs]
        return np.stack(states)
    def step(self, actions):
        results = [self.envs[i].step(actions[i]) for i in range(self.num_envs)]
        obs, rew, term, info = zip(*results)
        return np.stack(obs), np.array(rew), np.array(term), list(info)
    def state_vector(self):
        return np.stack([e.state_vector() for e in self.envs])
    def get_normalized_score(self, totals):
        return np.array([self.envs[i].get_normalized_score(totals[i]) for i in range(self.num_envs)])
    @property
    def unwrapped(self):
        return self.envs

#---------------------------------- setup ----------------------------------#

NUM_ENVS = 8
args = Parser().parse_args('plan')

# logger = utils.Logger(args)

env = MazeVectorEnv(NUM_ENVS, args.dataset)

print(" ---- Args ---- ")
print(args)
print(" -------------- ")

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(
    args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

#---------------------------------- main loop ----------------------------------#

observation = env.reset()

if args.conditional:
    print('Resetting target')
    for e in env.envs:
        e.set_target()

# set conditioning xy position to be the goal
targets = np.stack([e.unwrapped._target for e in env.envs])
cond = {
    diffusion.horizon - 1: np.hstack([targets, np.zeros((NUM_ENVS, 2))]),
}

# observations for rendering
rollout = [observation.copy()]

total_reward = np.zeros(NUM_ENVS)
for t in range(env.envs[0].max_episode_steps):

    state = env.state_vector().copy()

    # can replan if desired, but the open-loop plans are good enough for maze2d
    # that we really only need to plan once
    if t == 0:
        cond[0] = observation

        action, samples = policy(cond, batch_size=NUM_ENVS)
        actions = samples.actions
        sequence = samples.observations
    # pdb.set_trace()

    # ####
    if t < sequence.shape[1] - 1:
        next_waypoint = sequence[:, t+1]
    else:
        next_waypoint = sequence[:, -1].copy()
        next_waypoint[:, 2:] = 0
        # pdb.set_trace()

    # can use actions or define a simple controller based on state predictions
    action = (next_waypoint[:, :2] - state[:, :2]) + (next_waypoint[:, 2:] - state[:, 2:])
    # pdb.set_trace()
    ####

    next_observation, reward, terminal, _ = env.step(action)
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    for i in range(NUM_ENVS):
        if t == len(sequence) - 1:
            print(
                f'env {i} t: {t} | r: {reward[i]:.2f} |  R: {total_reward[i]:.2f} | '
                f'score: {score[i]:.4f} | {action[i]}'
            )

            if 'maze2d' in args.dataset:
                xy = next_observation[i, :2]
                goal = env.envs[i].unwrapped._target
                print(f'env {i} maze | pos: {xy} | goal: {goal}')

    # update rollout observations
    rollout.append(next_observation.copy())

    # logger.log(score=score, step=t)

    if t % args.vis_freq == 0 or terminal.any():
        fullpath = join(args.savepath, f'{t}.png')

        if t == 0:
            renderer.composite(fullpath, samples.observations, ncol=1)

        # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        # save rollout thus far
        renderer.composite(join(args.savepath, 'rollout.png'),
                           np.array(rollout)[None], ncol=1)

        # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        # logger.video(rollout=join(args.savepath, f'rollout.mp4'),
        #              plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

    if terminal.any():
        break

    observation = next_observation

# logger.finish(t, env.envs[0].max_episode_steps, score=score, value=0)

# save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {
    'score': score.tolist(),
    'step': int(t),
    'return': total_reward.tolist(),
    'term': terminal.tolist(),
    'epoch_diffusion': diffusion_experiment.epoch
}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

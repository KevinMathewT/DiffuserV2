import os
import math
import copy
import numpy as np
import torch
import einops
import pdb
from gym.vector import SyncVectorEnv

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
from diffuser.datasets.d4rl import load_environment

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        n_samples=2,
        bucket=None,
        n_train_steps=100000,
        warmup_steps=10000,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=8, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=8, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer

        # self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)
        self.optimizer = torch.optim.AdamW(
            diffusion_model.parameters(),
            lr=train_lr,
            weight_decay=1e-4,
            betas=(0.9,0.999)
        )
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, n_train_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scheduler = None

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, epoch, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)

                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()   
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                self.save(epoch, step)

            if self.step % self.log_freq == 0:
                lr = self.optimizer.param_groups[0]['lr']
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f} | lr: {lr:.12f}')

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                avg_rewards = self.render_samples(n_samples=self.n_samples)
                print(f'[ eval ] average reward at step {i}: {avg_rewards}')
                self.evaluate_batch_score() # new: evaluate a larger batch of rollouts and print avg score

            self.step += 1

    def save(self, epoch, step):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}_{step}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        # also show the start condition on the reference plots
        # unnormalize the single condition dict
        orig_cond = batch.conditions                    # {0: normed_obs}
        unnorm = {
            k: self.dataset.normalizer.unnormalize(v, 'observations')
            for k, v in orig_cond.items()
        }
        conds = [unnorm] * observations.shape[0]        # one per path row
        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(
            savepath,
            observations,
            conditions=conds
        )

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model.conditional_sample(conditions)
            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)

    
    def _mean_reward(self, samples):
        """
        Vectorized rollout of `samples` through B envs in parallel.
        """
        B, H, _ = samples.shape                                  # [B x H x T]
        fns = [lambda name=self.env.name: load_environment(name) for _ in range(B)]
        envs = SyncVectorEnv(fns)
        obs = envs.reset()                                       # [B x obs_dim]
        tot = np.zeros(B)                                        # [B]
        for t in range(H):
            acts = samples[:, t, :self.dataset.action_dim]       # [B x A]
            obs, r, _, _ = envs.step(acts)                       # obs: [B x obs_dim], r: [B]
            tot += r
        res = [
            e.get_normalized_score(v) if hasattr(e, "get_normalized_score") else v
            for e, v in zip(envs.envs, tot)
        ]                                                       # [B]
        envs.close()
        return sum(res) / B

    def evaluate_batch_score(self, B=32):
        """
        Vectorized rollout with detailed prints.
        """
        from diffuser.guides.policies import Policy
        # from diffuser.utils.serialization import load_diffusion
        import diffuser.datasets as datasets
        
        # diffusion_experiment = load_diffusion("", "", self.logdir)

        diffusion = self.ema_model
        dataset = self.dataset
        renderer = self.renderer

        policy = Policy(diffusion, self.dataset.normalizer)
        env = datasets.load_environment(self.dataset_name)

        total_batch_reward = 0
        total_batch_score = 0

        #---------------------------------- main loop ----------------------------------#
        for i in range(B):    
            observation = env.reset()

            ## set conditioning xy position to be the goal
            env.set_target()
            target = env._target
            cond = {
                diffusion.horizon - 1: np.array([*target, 0, 0]),
            }

            ## observations for rendering
            rollout = [observation.copy()]

            total_reward = 0
            for t in range(env.max_episode_steps):

                state = env.state_vector().copy()

                ## can replan if desired, but the open-loop plans are good enough for maze2d
                ## that we really only need to plan once
                if t == 0:
                    cond[0] = observation

                    action, samples = policy(cond, batch_size=1)
                    actions = samples.actions[0]
                    sequence = samples.observations[0]
                # pdb.set_trace()

                # ####
                if t < len(sequence) - 1:
                    next_waypoint = sequence[t+1]
                else:
                    next_waypoint = sequence[-1].copy()
                    next_waypoint[2:] = 0
                    # pdb.set_trace()

                ## can use actions or define a simple controller based on state predictions
                action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
                # pdb.set_trace()
                ####

                # else:
                #     actions = actions[1:]
                #     if len(actions) > 1:
                #         action = actions[0]
                #     else:
                #         # action = np.zeros(2)
                #         action = -state[2:]
                #         pdb.set_trace()



                next_observation, reward, terminal, _ = env.step(action)
                total_reward += reward
                score = env.get_normalized_score(total_reward)
                if t == env.max_episode_steps - 1:
                    print(
                        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
                        f'{action}'
                    )
                    total_batch_reward += total_reward
                    total_batch_score += score

                if 'maze2d' in self.dataset_name:
                    xy = next_observation[:2]
                    goal = env.unwrapped._target
                    if t == env.max_episode_steps - 1:
                        print(
                            f'maze | pos: {xy} | goal: {goal}'
                        )

                ## update rollout observations
                rollout.append(next_observation.copy())

                # logger.log(score=score, step=t)

                # if t % args.vis_freq == 0 or terminal:
                #     fullpath = join(args.savepath, f'{t}.png')

                    # if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


                    # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

                    ## save rollout thus far
                    # renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)

                    # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

                    # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

                if terminal:
                    break

                observation = next_observation

        print(f"Average batch reward: {total_batch_reward / B} | Average batch score: {total_batch_score / B}")
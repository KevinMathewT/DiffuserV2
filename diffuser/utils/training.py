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

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

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

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)

                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.scheduler.step()   
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

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

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
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
        """
        renders samples from (ema) diffusion model, marking start/end
        """
        rewards = []
        for i in range(batch_size):
            batch = self.dataloader_vis.__next__()
            # sample trajectories as before
            conds_t = to_device(batch.conditions, 'cuda:0')
            conds_t = apply_dict(einops.repeat, conds_t,
                                 'b d -> (repeat b) d', repeat=n_samples)
            samples = self.ema_model.conditional_sample(conds_t)
            samples = to_np(samples)

            # build observation tracks [n_samples x (H+1) x obs_dim]
            H = samples.shape[1]
            obs_dim = self.dataset.action_dim
            normed_obs = samples[:, :, obs_dim:]
            start = to_np(batch.conditions[0])[None]              # [1 x obs_dim]
            normed_obs = np.concatenate([np.repeat(start, n_samples, 0),
                                          normed_obs], axis=1)
            observations = self.dataset.normalizer.unnormalize(
                normed_obs, 'observations'
            )

            # --- NEW: derive start/end from the _same_ observations array ---
            # pick out x,y (first two dims)
            starts = observations[:, 0, :2]   # [n_samples x 2]
            ends   = observations[:, -1, :2]  # [n_samples x 2]
            conds = []
            for s, e in zip(starts, ends):
                conds.append({
                    0:   s,
                    H-1: e
                })

            savepath = os.path.join(
                self.logdir, f'sample-{self.step}-{i}.png'
            )
            self.renderer.composite(
                savepath,
                observations,
                conditions=conds
            )

            rewards.append(self._mean_reward(samples))

        return sum(rewards) / len(rewards)

    
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

    def evaluate_batch_score(self, B=128, ncol=4):
        """
        Vectorized rollout with detailed prints.
        """
        from diffuser.guides.policies import Policy
        
        fns  = [lambda name=self.env.name: load_environment(name) for _ in range(B)]
        envs = SyncVectorEnv(fns)
        obs0 = envs.reset()
        # set targets
        tg = [ (e.set_target() or e._target) if hasattr(e,'set_target') else e._target
               for e in envs.envs ]
        tg = np.stack(tg)
        cond = {
            0: obs0,
            self.model.horizon - 1:
                np.concatenate([tg, np.zeros((B,2))], axis=1)
        }
        _, traj = Policy(self.ema_model, self.dataset.normalizer)(
                       cond, batch_size=B)
        acts = traj.actions                         # [B x H x A]
        paths = [[o] for o in obs0]                 # list of B lists
        tot   = np.zeros(B)
        for t in range(acts.shape[1]):
            o, r, _, _ = envs.step(acts[:,t])       # o:[B x obs], r:[B]
            tot += r
            for i,x in enumerate(o): paths[i].append(x)
        res = [ e.get_normalized_score(v) if hasattr(e,'get_normalized_score') else v
                for e,v in zip(envs.envs, tot) ]
        envs.close()
        avg = sum(res) / B
        print(f'[ eval ] avg normalized score over {B}: {avg:.4f}')

        arr = np.array([np.array(p) for p in paths])            # [B x (H+1) x obs]
        conds = [{k: cond[k][i] for k in cond} for i in range(B)]
        self.renderer.composite(
            os.path.join(self.logdir, f'eval-rollouts-{self.step}.png'),
            arr,
            ncol=(ncol if B % ncol == 0 else B),
            conditions=conds
        )
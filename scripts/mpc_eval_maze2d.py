import json, copy
import numpy as np
from os.path import join
from diffuser.guides.policies import Policy
import diffuser.datasets as datasets, diffuser.utils as utils

class Parser(utils.Parser):
    dataset:       str   = 'maze2d-umaze-v1'
    config:        str   = 'config.maze2d'
    num_rollouts:  int   = 32
    num_samples:   int   = 16
    num_segments:  int   = 4
    temperature:   float = 4

def main():
    args = Parser().parse_args('plan')
    env   = datasets.load_environment(args.dataset)
    de    = utils.load_diffusion(
                args.logbase, args.dataset, args.diffusion_loadpath,
                epoch=args.diffusion_epoch
            )
    model  = de.ema
    policy = Policy(model, de.dataset.normalizer)
    horizon        = model.horizon
    segment_length = horizon // args.num_segments

    total_return = 0.0
    total_score  = 0.0

    for rollout_idx in range(args.num_rollouts):
        state = env.reset()
        if args.conditional:
            env.set_target()
        # save the very first start and the fixed goal
        start_state = state.copy()
        target      = env._target
        goal_obs    = np.array([*target, 0, 0])

        t              = 0
        episode_return = 0.0
        episode_score  = 0.0
        sequence       = None

        while t < env.max_episode_steps:
            if t % segment_length == 0 and t < horizon:
                # anchor start, current, and goal for inpainting
                cond = {
                    0:                start_state,
                    t:                state,
                    horizon - 1:      goal_obs,
                }
                _, out = policy(cond, batch_size=args.num_samples)
                trajs   = out.observations  # (num_samples, horizon, dim)

                # compute normalized returns for each sample
                scores = []
                for i in range(args.num_samples):
                    sim_env = copy.deepcopy(env)
                    sim_ret = 0.0
                    for k in range(horizon - 1):
                        wp = trajs[i][k+1]
                        s0 = sim_env.state_vector()
                        a  = wp[:2] - s0[:2] + (wp[2:] - s0[2:])
                        _, r_step, done, _ = sim_env.step(a)
                        sim_ret += r_step
                        if done:
                            break
                    scores.append(env.get_normalized_score(sim_ret))

                w = np.exp(np.array(scores) / args.temperature)
                w /= w.sum()
                sequence = (w[:, None, None] * trajs).sum(0)

            # execute next action
            s0 = env.state_vector()
            if t < len(sequence) - 1:
                wp = sequence[t+1]
            else:
                wp     = sequence[-1].copy()
                wp[2:] = 0
            action, next_state, reward, done, _ = (
                wp[:2] - s0[:2] + (wp[2:] - s0[2:]),
                *env.step(wp[:2] - s0[:2] + (wp[2:] - s0[2:]))
            )

            episode_return += reward
            episode_score  = env.get_normalized_score(episode_return)

            if done or t == env.max_episode_steps - 1:
                print(
                    f't:{t} r:{reward:.2f} R:{episode_return:.2f} '
                    f'score:{episode_score:.4f} a:{action}'
                )
            if done:
                break

            state = next_state
            t += 1

        total_return += episode_return
        total_score  += episode_score

    # save final results
    result = {
        'score':           episode_score,
        'step':            t,
        'return':          episode_return,
        'term':            done,
        'epoch_diffusion': de.epoch
    }
    with open(join(args.savepath, 'rollout.json'), 'w') as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(f'Average total reward over {args.num_rollouts} rollouts: '
          f'{total_return/args.num_rollouts:.2f}')
    print(f'Average score over {args.num_rollouts} rollouts: '
          f'{total_score/args.num_rollouts:.4f}')

if __name__ == '__main__':
    main()

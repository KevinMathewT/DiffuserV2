import json
import numpy as np
from os.path import join
import pdb
from joblib import Parallel, delayed

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

NUM_ENVS = args.batch_size   # run this many roll-outs in parallel

print(" ---- Args ---- ")
print(args)
print(" -------------- ")

#---------------------------------- global (light-weight) paths ------------#

LOGBASE   = args.logbase
DATANAME  = args.dataset
LOADPATH  = args.diffusion_loadpath
EPOCH     = args.diffusion_epoch
SAVEBASE  = args.savepath

#---------------------------------- worker ---------------------------------#

def run_single(idx):
    """
    *Create* heavy objects (env, diffusion model, policy) inside the worker.
    This avoids pickling errors with joblib.
    """
    env = datasets.load_environment(DATANAME)

    diff_exp = utils.load_diffusion(LOGBASE, DATANAME, LOADPATH, epoch=EPOCH)
    diffusion = diff_exp.ema
    policy    = Policy(diffusion, diff_exp.dataset.normalizer)

    observation = env.reset()

    print(f'env {idx} | conditional: {args.conditional}')
    if args.conditional:
        env.set_target()

    target = env._target
    cond   = {diffusion.horizon - 1: np.array([*target, 0, 0])}

    rollout      = [observation.copy()]
    total_reward = 0

    for t in range(env.max_episode_steps):
        state = env.state_vector().copy()

        if t == 0:
            cond[0]     = observation
            _, samples  = policy(cond, batch_size=1)
            sequence    = samples.observations[0]

        next_wp = sequence[t+1] if t < len(sequence)-1 else sequence[-1].copy()
        if t >= len(sequence)-1:
            next_wp[2:] = 0

        action = next_wp[:2] - state[:2] + (next_wp[2:] - state[2:])
        observation, reward, terminal, _ = env.step(action)

        total_reward += reward
        score = env.get_normalized_score(total_reward)
        
        if t == len(sequence) - 1:
            print(
                f'env {idx} | t: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | '
                f'score: {score:.4f} | {action}'
            )

        rollout.append(observation.copy())
        if terminal:
            break

    json_path = join(SAVEBASE, f'rollout_env{idx}.json')
    json.dump(
        {'score': score, 'step': t, 'return': total_reward, 'term': bool(terminal),
         'epoch_diffusion': diff_exp.epoch},
        open(json_path, 'w'), indent=2, sort_keys=True
    )
    return score

#---------------------------------- parallel run ---------------------------#

scores = Parallel(n_jobs=NUM_ENVS, backend='loky')(
    [delayed(run_single)(i) for i in range(NUM_ENVS)]
)

print(f'Average score over {NUM_ENVS} envs: {np.mean(scores):.4f}')

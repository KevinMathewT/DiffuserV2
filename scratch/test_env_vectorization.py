#!/usr/bin/env python3
import numpy as np
from gym.vector import SyncVectorEnv
from diffuser.datasets import load_environment

def mean_single(envn, samples):
    res = []
    for i, seq in enumerate(samples):
        e = load_environment(envn); e.seed(i); e.reset()
        tot = 0
        for a in seq:
            _, r, done, _ = e.step(a); tot += r
            if done: break
        if hasattr(e, 'get_normalized_score'):
            tot = e.get_normalized_score(tot)
        res.append(tot)
    return res

def mean_vec(envn, samples):
    B, H, _ = samples.shape
    def mk(i):
        def _init():
            e = load_environment(envn); e.seed(i); return e
        return _init
    envs = SyncVectorEnv([mk(i) for i in range(B)])
    envs.reset()
    tot = np.zeros(B)
    for t in range(H):
        _, r, _, _ = envs.step(samples[:, t]); tot += r
    res = []
    for e, v in zip(envs.envs, tot):
        if hasattr(e, 'get_normalized_score'):
            v = e.get_normalized_score(v)
        res.append(v)
    envs.close()
    return res

if __name__ == '__main__':
    env_name = 'maze2d-umaze-v1'
    env = load_environment(env_name)
    B = 4
    H = env.max_episode_steps
    A = env.action_space.shape[0]

    np.random.seed(0)
    lo, hi = env.action_space.low, env.action_space.high
    samples = np.random.uniform(lo, hi, (B, H, A))

    s = mean_single(env_name, samples)
    v = mean_vec(env_name, samples)

    print('single-env rewards:    ', s)
    print('vectorized-env rewards:', v)
    assert np.allclose(s, v), "❌ rewards mismatch!"
    print("✅ OK, both implementations produce identical rewards.")

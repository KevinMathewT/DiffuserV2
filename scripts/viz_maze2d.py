import os, json, imageio, numpy as np, torch
from os.path import join
from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils

# ---------- parser ----------
class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config:  str = 'config.maze2d'
args = Parser().parse_args('plan')

# ---------- env & diffusion ----------
env  = datasets.load_environment(args.dataset)
exp  = utils.load_diffusion(args.logbase, args.dataset,
                            args.diffusion_loadpath, epoch=args.diffusion_epoch)
diff, ds, rdr = exp.ema, exp.dataset, exp.renderer
pol  = Policy(diff, ds.normalizer)

# ---------- main rollout (exactly your “correct” version) ----------
obs = env.reset()
if args.conditional:
    print('Resetting target'); env.set_target()
goal = env._target
cond = {diff.horizon-1: np.array([*goal,0,0])}

roll, ret = [obs.copy()], 0
for t in range(env.max_episode_steps):
    st = env.state_vector().copy()
    if t == 0:
        cond[0] = obs
        _, smp = pol(cond, batch_size=args.batch_size)
        seq = smp.observations[0]

    nxt = seq[t+1] if t < len(seq)-1 else seq[-1].copy(); nxt[2:] = 0
    act = nxt[:2]-st[:2] + (nxt[2:]-st[2:])
    obs, r, term, _ = env.step(act)
    ret += r; sc = env.get_normalized_score(ret)
    print(f't:{t}|r:{r:.2f}|R:{ret:.2f}|score:{sc:.4f}|{act}')
    roll.append(obs.copy())

    if t % args.vis_freq == 0 or term:
        if t == 0: rdr.composite(join(args.savepath,f'{t}.png'), smp.observations, ncol=1)
        rdr.composite(join(args.savepath,'rollout.png'), np.array(roll)[None], ncol=1)
    if term: break

json.dump({'score':sc,'step':t,'return':ret,'term':term,'epoch_diffusion':exp.epoch},
          open(join(args.savepath,'rollout.json'),'w'), indent=2)

# ---------- diffusion‑backward high‑res video ----------
def diff_video(idx=0, fps=10):
    batch=ds[idx]
    dev  = diff.betas.device
    c    = {k: torch.tensor(v[None], device=dev, dtype=torch.float32)
            for k, v in batch.conditions.items()}
    _, D = diff.conditional_sample(c, return_diffusion=True)   # (1,T+1,H,Tr)
    D = D[0].cpu().numpy(); start = batch.conditions[0][None]
    frames=[]
    for i, step in enumerate(D):
        traj = np.concatenate([start, step[:,ds.action_dim:]], 0)[None]
        obs  = ds.normalizer.unnormalize(traj,'observations')
        img  = join(args.savepath, f'diff_{i}.png')
        rdr.composite(img, obs, ncol=1, label=[f"t = {i}"])
        frames.append(imageio.imread(img))
    with imageio.get_writer(join(args.savepath,'diffusion.mp4'),
                            fps=fps, macro_block_size=None) as w:
        for f in frames: w.append_data(f)

diff_video()
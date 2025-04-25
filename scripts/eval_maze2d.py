# execute: python -m scripts.eval_maze2d   --dataset maze2d-umaze-v1   --logbase /home/km6748/DiffuserV2/pretrained/logs   --diffusion_loadpath diffusion/H128_T64   --diffusion_epoch 1760000   --batch_size 128   --vis_freq 10   --conditional True

import json, numpy as np
from os.path import join
from diffuser.guides.policies import Policy
import diffuser.datasets as ds
import diffuser.utils as ut

class Parser(ut.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'

args = Parser().parse_args('plan')
N=512  # parallel envs
USE_MODEL_ACTION=True  # toggle between model actions vs waypoint-following

exp=ut.load_diffusion(args.logbase,args.dataset,args.diffusion_loadpath,epoch=args.diffusion_epoch)
ema,dset,rnd=exp.ema,exp.dataset,exp.renderer
pol=Policy(ema,dset.normalizer)
H=ema.horizon

es=[ds.load_environment(args.dataset) for _ in range(N)]
obs=[e.reset() for e in es]
if args.conditional:
    for e in es:e.set_target()

actions=[None]*N
seqs=[None]*N
for i,e in enumerate(es):
    trg=[*e._target,0,0]
    cd={H-1:np.array(trg),0:obs[i]}
    _,smp=pol(cd,batch_size=args.batch_size)
    actions[i]=smp.actions[0]      # [H, act_dim]
    seqs[i]=smp.observations[0]    # [H, obs_dim]

rol=[[o.copy()] for o in obs]
ret=np.zeros(N)
sc=np.zeros(N)
done=np.zeros(N,bool)

for t in range(es[0].max_episode_steps):
    rs=[]
    for i,e in enumerate(es):
        if done[i]:
            rs.append(0);continue
        state=e.state_vector().copy()
        if USE_MODEL_ACTION:
            a=actions[i][t] if t<len(actions[i]) else np.zeros(e.action_space.shape)
        else:
            if t<len(seqs[i])-1:
                wp=seqs[i][t+1]
            else:
                wp=seqs[i][-1].copy();wp[2:]=0
            a=wp[:2]-state[:2]+(wp[2:]-state[2:])
        o2,r,term,_=e.step(a)
        ret[i]+=r
        sc[i]=e.get_normalized_score(ret[i])
        rol[i].append(o2.copy())
        obs[i]=o2
        if term:done[i]=True
        rs.append(r)
    print(f't:{t} | r_avg:{np.mean(rs):.2f} | R_avg:{ret.mean():.2f} | score_avg:{sc.mean():.4f}')
    if t%args.vis_freq==0:
        rnd.composite(join(args.savepath,f'{t}.png'),np.array(rol[0])[None],ncol=1)
        rnd.composite(join(args.savepath,'rollout.png'),np.array(rol[0])[None],ncol=1)
    if done.all():break

print(f'FINAL | steps:{t} | R_avg:{ret.mean():.2f} | score_avg:{sc.mean():.4f}')

json.dump({'score_avg':float(sc.mean()),'return_avg':float(ret.mean()),'step':int(t),'term':bool(done.all()),'epoch_diffusion':exp.epoch},open(join(args.savepath,'rollout.json'),'w'),indent=2,sort_keys=True)

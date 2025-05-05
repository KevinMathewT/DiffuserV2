import pdb
import torch
import glob
import re
from time import time

import diffuser.utils as utils
import diffuser.datasets as datasets

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'
    loadpath: str = None

args = Parser().parse_args('diffusion')


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    # predict_epsilon=args.predict_epsilon,
    parameterization=args.parameterization,    # added
    v_posterior=args.v_posterior,              # added
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
    n_train_steps=args.n_train_steps,
    warmup_steps=int(0.1 * args.n_train_steps),
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()
diffusion = diffusion_config(model)

# to ignore buffer values
if args.loadpath:
    ck = max(
        glob.glob(f"{args.loadpath}/state_*.pt"),
        key=lambda p: int(re.search(r"state_(\d+)\.pt", p).group(1))
    )
    ckpt = torch.load(ck, map_location=args.device)
    ema_dict = ckpt['ema']
    current_state = diffusion.state_dict()
    trainable = { name for name, _ in diffusion.named_parameters() }
    to_load = {
        k: v
        for k, v in ema_dict.items()
        if k in trainable and v.shape == current_state[k].shape
    }

    print(f"will load {len(to_load)} / {len(trainable)} trainable params")
    missing, unexpected = diffusion.load_state_dict(to_load, strict=False)
    print("loaded → missing keys:", missing)
    print("           unexpected keys:", unexpected)
    print(f"✅  EMA parameters restored from {ck}")

# to load all weights
# if args.loadpath:
#     ck = max(
#         glob.glob(f"{args.loadpath}/state_*.pt"),
#         key=lambda p: int(re.search(r"state_(\d+)\.pt", p).group(1))
#     )
#     ema = torch.load(ck, map_location=args.device)["ema"]

#     # intersection: only keys that the current diffusion actually has
#     state_keys = diffusion.state_dict().keys()
#     to_load    = {k: v for k, v in ema.items() if k in state_keys}

#     print(f">>> loading {len(to_load)}/{len(state_keys)} entries (params + buffers)")
#     missing, unexpected = diffusion.load_state_dict(to_load, strict=False)
#     print("missing :", missing)        # usually []
#     print("unexpected:", unexpected)   # usually []
#     print(f"✅  EMA parameters *and* buffers restored from {ck}")

print("=== diffusion model ===")
print(diffusion)
for name, buf in diffusion.named_buffers():
    print(f"{name:30s}  shape={tuple(buf.shape)}  mean={buf.mean():.6f}  std={buf.std():.6f}")
print("=========================")

trainer = trainer_config(diffusion, dataset, renderer)
trainer.env = datasets.load_environment(args.dataset) # attach env so Trainer.evaluate_batch_score can use it
trainer.dataset_name = args.dataset
trainer.evaluate_batch_score()


#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

start = time()
for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(i, n_train_steps=args.n_steps_per_epoch)
    print(f'Epoch {i} | epoch time: {time() - start:.2f}s')


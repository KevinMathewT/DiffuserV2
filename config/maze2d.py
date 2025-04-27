import socket
import datetime
from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args
now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('parameterization', 'P'),            # added v-pred parameterization
    ('v_posterior', 'v'),                 # added v-pred posterior weight
    ('time', 't'),                        # time
]


plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
]

base = {

    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 256,
        'n_diffusion_steps': 256,
        'parameterization': 'eps',           # default parameterization
        'v_posterior': 0.0,                 # default v-posterior weight
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.Maze2dRenderer',

        ## dataset
        'loader': 'datasets.GoalDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        # 'n_train_steps': 2e6,              # ################ #
        # 'batch_size': 32,                  #  original params #
        # 'learning_rate': 2e-4,             #                  #
        # 'gradient_accumulate_every': 2,    # ################ #
        'n_train_steps': 2e6,          
        'batch_size': 64,              
        'learning_rate': 2e-4,         
        'gradient_accumulate_every': 1,
        # 'n_train_steps': 500000, # /4
        # 'batch_size': 256, # 8× 
        # 'learning_rate': 8e-4, # 4×
        # 'gradient_accumulate_every': 1, # /2
        # 'n_train_steps': 250000, # /8
        # 'batch_size': 512, # 16× 
        # 'learning_rate': 1.6e-3, # 8×
        # 'gradient_accumulate_every': 1, # /2
        # 'n_train_steps': 125000, # /8
        # 'batch_size': 1024, # 32× 
        # 'learning_rate': 1.6e-3, # 8× (technically 16×)
        # 'gradient_accumulate_every': 1, # /2
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 1000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda',

        ## time
        'time': now,  # added timestamp
    },

    'plan': {
        'batch_size': 1,
        'device': 'cuda',

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        'conditional': False,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },

}

#------------------------ overrides ------------------------#

'''
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
'''

maze2d_umaze_v1 = {
    'diffusion': {
        'horizon': 128,
        'n_diffusion_steps': 64,
        # 'parameterization': 'v',            # override to v-pred
        # 'v_posterior': 0.5,                # example non-zero weight
    },
    'plan': {
        'horizon': 128,
        'n_diffusion_steps': 64,
    },
}

maze2d_large_v1 = {
    'diffusion': {
        'horizon': 384,
        'n_diffusion_steps': 256,
        # 'parameterization': 'v',            # override to v-pred
        # 'v_posterior': 0.5,                # example non-zero weight
    },
    'plan': {
        'horizon': 384,
        'n_diffusion_steps': 256,
    },
}

import sys
import os

sys.path.append(os.path.abspath("libs"))

import torch
from tqdm.auto import tqdm

from point_e.point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.point_e.diffusion.sampler import PointCloudSampler
from point_e.point_e.models.download import load_checkpoint
from point_e.point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.point_e.util.plotting import plot_point_cloud

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating base model...')
base_name = 'base40M-textvec'
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print('creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print('downloading base checkpoint...')
base_model.load_state_dict(load_checkpoint(base_name, device))

print('downloading upsampler checkpoint...')
upsampler_model.load_state_dict(load_checkpoint('upsample', device))
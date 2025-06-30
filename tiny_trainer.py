import math
import os
import random
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from PIL import Image
from torch import nn
from tqdm import tqdm
from typing_extensions import Literal, assert_never
from utils import knn, rgb_to_sh

from gsplat import export_splats
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 10,
    scales_lr: float = 5e-2,
    opacities_lr: float = 5e-1,
    quats_lr: float = 1e-2,
    sh0_lr: float = 2.5e-2,
    shN_lr: float = 2.5e-2 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    batch_size: int = 1,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
    rgbs = torch.rand((init_num_pts, 3))

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
    colors[:, 0, :] = rgb_to_sh(rgbs)
    params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
    params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))

    # color is SH coefficients.
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
    colors[:, 0, :] = rgb_to_sh(rgbs)
    params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
    params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size), "name": name}],
            eps=1e-15 / math.sqrt(batch_size),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


def initialize_stable_diffusion(
    model: str = "sd-legacy/stable-diffusion-v1-5",
    num_steps: int = 1000,
    device: str = "cuda",
    prompt: str = "A photo of an hamburger",
):
    pipe = StableDiffusionPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None,
    ).to(device)

    vae = pipe.vae.eval()
    unet = pipe.unet.eval()
    text_encoder = pipe.text_encoder.eval()
    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps)
    scheduler.timesteps = scheduler.timesteps.to(device)

    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        text_embeds = text_encoder(**text_inputs).last_hidden_state

    null_inputs = tokenizer(
        [""],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        null_embeds = text_encoder(**null_inputs).last_hidden_state

    alphas = scheduler.alphas_cumprod.to(device)

    return vae, unet, scheduler, text_embeds, null_embeds, alphas


def initialize_cameras(radius, n_camera=100, device="cuda"):
    up = torch.tensor([0.0, 1.0, 0.0])

    target = torch.asarray([0, 0, 0], device=device)
    angles = torch.linspace(-math.pi / 3, math.pi / 3, n_camera + 1)[:-1]

    origins = torch.stack(
        [
            radius * torch.sin(angles),  # X varie de gauche à droite
            torch.full((n_camera,), target[1]),  # Y fixe (hauteur)
            radius * torch.cos(angles),  # Z varie pour faire le cercle
        ],
        dim=1,
    ).to(device)

    # origins: (n_camera, 3), target: (3,)
    forwards = nn.functional.normalize(
        target.unsqueeze(0) - origins, dim=1
    )  # (n_camera, 3)
    up = up.to(forwards.device)
    rights = nn.functional.normalize(
        torch.cross(up.expand(n_camera, -1), forwards, dim=1), dim=1
    )
    ups = torch.cross(forwards, rights, dim=1)
    rot = torch.stack([rights, ups, forwards], dim=2)  # (n_camera, 3, 3)
    camtoworld = (
        torch.eye(4, device=device).unsqueeze(0).repeat(n_camera, 1, 1)
    )  # (n_camera, 4, 4)
    camtoworld[:, :3, :3] = rot
    camtoworld[:, :3, 3] = origins
    return camtoworld


def get_sds_loss(latents, text_embeddings, alphas, guidance_scale=100, grad_scale=1.0):
    t = torch.randint(1, len(alphas) + 1, (1,), device=latents.device)
    z_t = torch.randn_like(latents)

    alpha_t = alphas[t - 1].view(-1, 1, 1, 1)
    xt = torch.sqrt(alpha_t) * latents + torch.sqrt(1 - alpha_t) * z_t

    latent_in = xt.repeat(2, 1, 1, 1)
    cond_uncond_embed = torch.cat([null_embeds, text_embeddings], dim=0)

    noise_pred = unet(
        latent_in, t.float(), encoder_hidden_states=cond_uncond_embed
    ).sample
    noise_uncond, noise_cond = noise_pred.chunk(2)

    noise_pred_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    grad = 2 * (noise_pred_guided - z_t)
    target = (latents - grad).detach()

    return nn.functional.mse_loss(latents, target) * grad_scale

def compute_loss(
    sds_loss,
    compactness_reg,
    opacity_reg,
    scale_reg,
    quat_reg,
    colors_reg,
    step: int,
    max_steps: int
):
    # Convert to torch float for computation
    step = torch.tensor(step, dtype=torch.float32)
    max_steps = torch.tensor(max_steps, dtype=torch.float32)
    progress = step / max_steps

    # Schedulers (modifiable selon le besoin)
    compactness_weight = 0 #(progress ** 2) * 1.0             # quadratic ramp-up
    opacity_weight = (1.0 - progress) * 5e-2               # linear ramp-down
    scale_weight = (progress ** 2) * 10.0                         # linear ramp-up
    quat_weight = (1.0 - progress) * 1e-3                  # linear ramp-down
    colors_weight = torch.sqrt(progress)                   # sqrt ramp-up

    # Total loss
    loss = sds_loss
    loss = loss + compactness_weight * compactness_reg
    loss = loss + opacity_weight * opacity_reg
    loss = loss + scale_weight * scale_reg
    loss = loss + quat_weight * quat_reg
    loss = loss + colors_weight * colors_reg

    return loss

cfg = Config(strategy=DefaultStrategy(verbose=True))
cfg.adjust_steps(cfg.steps_scaler)

device = "cuda"
save_dir = "save"
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)

splats, optimizers = create_splats_with_optimizers(
    init_num_pts=cfg.init_num_pts,
    init_extent=cfg.init_extent,
    init_opacity=cfg.init_opa,
    init_scale=cfg.init_scale,
    means_lr=cfg.means_lr * 5,
    scales_lr=cfg.scales_lr / 10,
    opacities_lr=cfg.opacities_lr,
    quats_lr=cfg.quats_lr,
    sh0_lr=cfg.sh0_lr,
    shN_lr=cfg.shN_lr,
    scene_scale=1,
    sh_degree=cfg.sh_degree,
    batch_size=cfg.batch_size,
    device=device,
)


cfg.strategy.check_sanity(splats, optimizers)

strategy_state = cfg.strategy.initialize_state(scene_scale=1)

# Create stable diffusion object
vae, unet, scheduler, text_embeds, null_embeds, alphas = initialize_stable_diffusion()

# Create cameras in the scene
width = 512
height = 512

fx = fy = 500.0
cx = width / 2
cy = height / 2

n_camera = 100

cams_to_world = initialize_cameras(radius=15, n_camera=n_camera, device=device)
Ks = (
    torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device)
    .unsqueeze(0)
    .repeat(n_camera, 1, 1)
)

guidance_scale = 25

# Train

# Init Train

num_steps = 5000

schedulers = [
    # means has a learning rate schedule, that end at 0.01 of the initial value
    torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / num_steps)
    ),
]

for step in tqdm(range(num_steps)):
    # Choose a camera from cams_to_world
    idx = torch.randint(0, n_camera, (1,)).item()
    camtoworld_selected = cams_to_world[idx % n_camera].unsqueeze(0)
    Ks_selected = Ks[idx % n_camera].unsqueeze(0)

    sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

    render_colors, render_alphas, info = rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=torch.cat([splats["sh0"], splats["shN"]], 1),
        viewmats=torch.linalg.inv(camtoworld_selected),  # [C, 4, 4]
        Ks=Ks_selected,  # [C, 3, 3]
        width=width,
        height=height,
        packed=cfg.packed,
        absgrad=(
            cfg.strategy.absgrad if isinstance(cfg.strategy, DefaultStrategy) else False
        ),
        sparse_grad=cfg.sparse_grad,
        rasterize_mode="classic",
        distributed=False,
        camera_model="pinhole",
        with_ut=cfg.with_ut,
        with_eval3d=cfg.with_eval3d,
        sh_degree=sh_degree_to_use,
    )

    # if random.random() < 0.5:
    #     bkgd = torch.tensor([0.0, 0.0, 0.0], device=device)
    # else:
    #     bkgd = torch.tensor([1.0, 1.0, 1.0], device=device)

    bkgd = torch.tensor([1.0, 1.0, 1.0], device=device)

    colors = render_colors + bkgd * (1.0 - render_alphas)
    colors = colors.permute(0, 3, 1, 2).clamp(0, 1) * 2 - 1
    # colors = F.interpolate(colors, size=(512, 512), mode='bilinear', align_corners=False)

    cfg.strategy.step_pre_backward(
        params=splats,
        optimizers=optimizers,
        state=strategy_state,
        step=step,
        info=info,
    )

    latent = vae.encode(colors).latent_dist.sample() * 0.18215
    sds_loss = get_sds_loss(latent, text_embeds, alphas, guidance_scale=guidance_scale)
    compactness_reg = splats["means"].norm(p=2, dim=1).mean()
    opacity_reg = ((torch.sigmoid(splats["opacities"]) - 0.5).abs()).mean()
    scale_reg = splats["scales"].norm(dim=1).mean()
    quat_reg = ((splats["quats"].norm(dim=1) - 1) ** 2).mean()
    colors_reg = torch.mean(colors**2)

    loss = compute_loss(sds_loss, compactness_reg, opacity_reg, scale_reg, quat_reg, colors_reg, step, num_steps)

    loss.backward()

    for k, optimizer in optimizers.items():
        optimizer.step()

    # for scheduler in schedulers:
    #     scheduler.step()

    cfg.strategy.step_post_backward(
        params=splats,
        optimizers=optimizers,
        state=strategy_state,
        step=step,
        info=info,
        packed=cfg.packed,
    )

    if step % 10 == 0:
        print(f"[{step:04d}] Loss: {loss.item():.4f}")
        print(
            f"Reg[step {step}] - SDS: {sds_loss.item():.4f} | Compact: {compactness_reg.item():.4f} "
            f"Opacity: {opacity_reg.item():.4f} | Scale: {scale_reg.item():.4f} | Quat: {quat_reg.item():.6f} | Colors: {colors_reg.item():.6f}"
        )

    if step % 1 == 0 or step == num_steps - 1:
        with torch.no_grad():
            decoded = vae.decode(latent / 0.18215).sample
            img = (decoded.clamp(-1, 1) + 1) / 2
            img = img[0].cpu().permute(1, 2, 0).numpy() * 255
            img = Image.fromarray(img.astype("uint8"))
            img.save(os.path.join(save_dir, f"step_{step:04d}.png"))

    if step % 50 == 0 or step == 0:
        export_splats(
            means=splats["means"],
            scales=splats["scales"],
            quats=splats["quats"],
            opacities=splats["opacities"],
            sh0=splats["sh0"],
            shN=splats["shN"],
            format="ply",
            save_to=f"save/point_cloud_{step}.ply",
        )

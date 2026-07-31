"""
Nicholas M. Boffi
10/5/25

Algorithmic comparison on the AFHQ-64 dataset.
"""

import os
import ml_collections

# Define experiments as tuples: (loss_type, psd_type, stopgrad_type)
experiments = [
    ("lsd", None, "convex"),
    ("psd", "uniform", "convex"),
    ("psd", "midpoint", "convex"),
    ("esd", None, "full"),
]


def get_config(
    slurm_id: int, dataset_location: str, output_folder: str
) -> ml_collections.ConfigDict:
    # ensure jax.device_count works (weird issue with importlib)
    import jax

    # Get experiment parameters
    loss_type, psd_type, stopgrad_type = experiments[slurm_id % len(experiments)]

    # setup overall config
    config = ml_collections.ConfigDict()

    # training config
    config.training = ml_collections.ConfigDict()
    config.training.shuffle = True
    config.training.conditional = False
    config.training.class_dropout = 0.0
    config.training.stopgrad_type = stopgrad_type
    config.training.psd_type = psd_type
    config.training.loss_type = loss_type
    config.training.tmin = 0.0
    config.training.tmax = 1.0
    config.training.seed = 42
    config.training.ema_facs = [0.999, 0.9999]
    config.training.ndevices = jax.device_count()

    # problem config - AFHQ64 specific
    config.problem = ml_collections.ConfigDict()
    config.problem.n = 15803
    config.problem.image_dims = (3, 64, 64)
    config.problem.d = 12288  # 3 * 64 * 64
    config.problem.num_classes = 0
    config.problem.target = "afhq64"
    config.problem.dataset_location = dataset_location
    config.problem.interp_type = "linear"
    config.problem.base = "gaussian"
    config.problem.gaussian_scale = "adaptive"

    # optimization config - Matching CelebA64 since same resolution
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 256  # Same as CelebA64
    config.optimization.diag_fraction = 0.75
    config.optimization.learning_rate = 1e-2  # Initial learning rate
    config.optimization.clip = 1.0
    config.optimization.total_samples = 204_800_000
    config.optimization.total_steps = int(
        config.optimization.total_samples // config.optimization.bs
    )
    config.optimization.decay_steps = 35000
    config.optimization.schedule_type = "sqrt"  # Square root schedule

    # logging config
    config.logging = ml_collections.ConfigDict()
    config.logging.plot_bs = 25
    config.logging.visual_freq = 5000
    config.logging.save_freq = 5000  # Save every 5k steps
    config.logging.wandb_project = "self-distill-flow-maps"

    # Create systematic name for the experiment
    method_str = f"{loss_type}_{psd_type}" if psd_type else loss_type

    config.logging.wandb_name = f"afhq64_paper_{method_str}"
    config.logging.wandb_entity = os.getenv("WANDB_ENTITY", "your-username")
    config.logging.output_folder = output_folder
    config.logging.output_name = config.logging.wandb_name

    # FID computation settings for AFHQ64
    config.logging.fid_freq = 10000  # Compute FID every 10k steps
    config.logging.fid_stats_path = f"{dataset_location}/afhq64/afhq64_fid_stats.npz"
    config.logging.fid_n_samples = 10000
    config.logging.fid_batch_size = 256
    config.logging.fid_n_steps_flow = [1, 2, 4, 8, 16]
    config.logging.fid_ema_factor = 0.9999
    config.logging.visual_ema_factor = 0.9999

    # network config
    config.network = ml_collections.ConfigDict()
    config.network.network_type = "edm2"
    config.network.load_path = ""  # No pretrained model
    config.network.img_resolution = config.problem.image_dims[1]
    config.network.img_channels = config.problem.image_dims[0]
    config.network.input_dims = config.problem.image_dims
    config.network.label_dim = 0  # No class conditioning for AFHQ
    config.network.use_cfg = False
    config.network.reset_optimizer = False
    config.network.logvar_channels = 128
    config.network.use_bfloat16 = True
    config.network.use_weight = True
    config.network.rescale = 0.5

    # CelebA64 architecture - appropriate for 64x64 resolution
    config.network.unet_kwargs = {
        "model_channels": 128,
        "channel_mult": [1, 2, 3, 4],
        "num_blocks": 3,
        "attn_resolutions": [16, 8],
        "block_kwargs": {
            "dropout": 0.0,  # No dropout for AFHQ
        },
    }

    return config

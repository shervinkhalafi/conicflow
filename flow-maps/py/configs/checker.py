"""
Nicholas M. Boffi
10/5/25

Algorithmic comparison on the two-dimensional checker dataset.
"""

import os
import ml_collections

# Define experiments matching the CelebA/CIFAR sweep
experiments = [
    ("lsd", None, "convex"),
    ("psd", "uniform", "convex"),
    ("psd", "midpoint", "convex"),
    ("esd", None, "full"),
]


def get_config(
    slurm_id: int, dataset_location: str = "", output_folder: str = ""
) -> ml_collections.ConfigDict:
    # ensure jax.device_count works (weird issue with importlib)
    import jax

    del dataset_location  # Not needed for checker

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
    config.training.ema_facs = [0.999]
    config.training.ndevices = jax.device_count()

    # problem config - Checker specific
    config.problem = ml_collections.ConfigDict()
    config.problem.n = 500_000  # 10M samples for checker
    config.problem.d = 2
    config.problem.image_dims = None    
    config.problem.num_classes = None
    config.problem.target = "checker"
    config.problem.dataset_location = None
    config.problem.interp_type = "linear"
    config.problem.base = "gaussian"
    config.problem.gaussian_scale = "adaptive"

    # optimization config - Using latest hyperparameters with checker-appropriate batch size
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 10_000  # Large batch for efficient 2D training
    config.optimization.diag_fraction = (
        0.75  # 75% diagonal fraction (matching latest configs)
    )
    config.optimization.learning_rate = 1e-3  # Standard for checker
    config.optimization.clip = 10.0
    config.optimization.total_steps = 250_000
    config.optimization.total_samples = (
        config.optimization.bs * config.optimization.total_steps
    )
    config.optimization.decay_steps = 2_000
    config.optimization.schedule_type = "sqrt"  # Square root schedule

    # logging config
    config.logging = ml_collections.ConfigDict()
    config.logging.plot_bs = 5000
    config.logging.visual_freq = 500
    config.logging.save_freq = 1000  # Save every 1k steps
    config.logging.wandb_project = "self-distill-flow-maps"

    # Create systematic name for the experiment
    method_str = f"{loss_type}_{psd_type}" if psd_type else loss_type

    config.logging.wandb_name = f"checker_paper_{method_str}"
    config.logging.wandb_entity = os.getenv("WANDB_ENTITY", "your-username")
    config.logging.output_folder = output_folder
    config.logging.output_name = config.logging.wandb_name

    # FID not relevant for checker
    config.logging.fid_freq = 0
    config.logging.fid_stats_path = None
    config.logging.fid_n_samples = None
    config.logging.fid_batch_size = None
    config.logging.fid_n_steps_flow = None
    config.logging.fid_ema_factor = None
    config.logging.visual_ema_factor = None

    # network config - 4-layer 512-neuron MLP (standard for checker)
    config.network = ml_collections.ConfigDict()
    config.network.network_type = "mlp"
    config.network.n_hidden = 3 # 3 hidden layers
    config.network.n_neurons = 256  # 256 neurons per layer
    config.network.output_dim = 2
    config.network.act = "gelu"
    config.network.use_residual = False
    config.network.use_weight = False
    config.network.use_bfloat16 = False
    config.network.rescale = 0.5  # sigma_data

    # Required but not used for MLP
    config.network.load_path = ""
    config.network.input_dims = (2,)
    config.network.load_ema_fac = None
    config.network.img_resolution = None
    config.network.img_channels = None
    config.network.label_dim = None
    config.network.logvar_channels = None
    config.network.reset_optimizer = True
    config.network.unet_kwargs = None

    return config

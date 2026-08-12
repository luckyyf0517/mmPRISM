import deepspeed
from deepspeed.accelerator import get_accelerator
import torch.distributed as dist

def get_train_ds_config(args):
    """Get DeepSpeed training configuration"""
    device = "cpu" if args.offload else "none"
    dtype_config = {"enabled": False}

    # Set data type
    if args.dtype == "fp16":
        data_type = "fp16"
        dtype_config = {"enabled": True, "loss_scale_window": 100}
    elif args.dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}
    elif args.dtype == "fp32":
        data_type = "fp32"
        dtype_config = {"enabled": True}
    else: 
        raise ValueError(f"Unsupported data type: {args.dtype}")

    # ZeRO optimization configuration
    zero_opt_dict = {
        "stage": args.zero_stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }

    return {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": args.gradient_clipping,
        "wall_clock_breakdown": False,
        
        # Mixed precision training settings
        "fp16": {
            "enabled": args.dtype == "fp16",
            "loss_scale_window": 100
        },
        "bf16": {
            "enabled": args.dtype == "bf16"
        }
    }


def add_deepspeed_args(parser):
    """Add DeepSpeed related command line arguments"""
    group = parser.add_argument_group('DeepSpeed', 'DeepSpeed configurations')
    
    # Add distributed training related arguments
    group.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')
    group.add_argument('--dist_backend', default='nccl', type=str, help='Distributed backend')
    group.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    
    # Existing DeepSpeed arguments
    group.add_argument('--offload',
                      action='store_true',
                      help='Enable ZeRO Offload techniques')
    
    group.add_argument('--dtype',
                      type=str,
                      default='fp32',
                      help='Training data type')
    
    group.add_argument('--zero_stage',
                      type=int,
                      default=2,
                      help='ZeRO optimization stage')
    
    group.add_argument('--gradient_accumulation_steps',
                      type=int,
                      default=1,
                      help='Number of gradient accumulation steps')
    
    group.add_argument('--gradient_clipping',
                      type=float,
                      default=1.0,
                      help='Gradient clipping value')
    
    return parser 
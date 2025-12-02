import numpy as np
import torch

from transformers import AutoModelForCausalLM
from accelerate.hooks import ModelHook, add_hook_to_module

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all

from lerobot.policies import (  # noqa: F401
    SmolVLAConfig,
)

class ZeroHook(ModelHook):
    def pre_forward(self, module, *args, **kwargs):
        print(f"Pre-forward hook for module: {type(module).__name__}")
        print(f"Input args: {args}")
        print(f"Input kwargs: {kwargs}")
        # You can modify args or kwargs here if needed
        return args, kwargs

    def post_forward(self, module, output):
        print(f"Post-forward hook for module: {type(module).__name__}")
        print(f"Output: {output.shape}")
        # You can modify the output here if needed
        output = np.zeros_like(output)
        return output

cfg = SmolVLAConfig.from_pretrained("adalal/smolvla-libero-object")
print(cfg.keys())
device = get_safe_torch_device(cfg.policy.device, log=True)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
set_seed(cfg.seed)

logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

logging.info("Making environment.")
envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

policy = SmolVLAPolicy.from_pretrained("adalal/smolvla-libero-object")
zero_hook = ZeroHook()

print(f"model before: {policy}")
add_hook_to_module(policy, zero_hook)
print(f"after: {policy}")

policy.eval()

processors = make_smolvla_pre_post_processors(
    config=policy_cfg,
    dataset_stats=kwargs.get("dataset_stats"),
)

with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
    info = eval_policy_all(
        envs=envs,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=cfg.eval.n_episodes,
        max_episodes_rendered=10,
        videos_dir=Path(cfg.output_dir) / "videos",
        start_seed=cfg.seed,
        max_parallel_tasks=cfg.env.max_parallel_tasks,
    )
    print("Overall Aggregated Metrics:")
    print(info["overall"])

    # Print per-suite stats
    for task_group, task_group_info in info.items():
        print(f"\nAggregated Metrics for {task_group}:")
        print(task_group_info)
# Close all vec envs
close_envs(envs)

# Save info
with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
    json.dump(info, f, indent=2)

logging.info("End of eval")

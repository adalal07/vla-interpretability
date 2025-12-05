import subprocess
import os
from hydra import main
from omegaconf import DictConfig, OmegaConf

@main(config_path="conf", config_name="config")
def run(cfg: DictConfig):

    print("Running config:")
    print(OmegaConf.to_yaml(cfg))

    # Create output directory for this config
    # out_dir = f"runs/{cfg.sweep_name}/" \
    #           f"target={cfg.target}_k={cfg.k}_alpha={cfg.alpha}_tk={cfg.top_k_tokens}"
    # os.makedirs(out_dir, exist_ok=True)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ##########################################
    # # 1. RUN find_value_vectors.py
    # ##########################################
    # cmd1 = [
    #     "python",
    #     f"{SCRIPT_DIR}/find_value_vectors.py",
    #     "--model", cfg.model,
    #     "--target", cfg.target,
    #     "--k", str(cfg.k),
    #     "--alpha", str(cfg.alpha),
    #     "--top-k-tokens", str(cfg.top_k_tokens),
    # ]

    # subprocess.run(cmd1, check=True)

    # ##########################################
    # # 2. RUN lerobot-eval
    # ##########################################
    # cmd2 = [
    #     "lerobot-eval",
    #     f"--policy.path={cfg.policy.path}",
    #     f"--policy.n_action_steps={cfg.policy.n_action_steps}",
    #     "--env.type=libero",
    #     "--env.task=libero_object",
    #     f"--eval.batch_size={cfg.eval.batch_size}",
    #     f"--eval.n_episodes={cfg.eval.n_episodes}",
    #     f"--output_dir={out_dir}",
    #     # f"--eval.condition_label={cfg.eval.condition_label}",
    # ]

    # subprocess.run(cmd2, check=True)

    # X VLA
    cmd = [
        "lerobot-eval" ,
        f"--policy.path=lerobot/xvla-libero" ,
        f"--policy.n_action_steps=10" ,
        f"--policy.auto_patch_target={cfg.target}" ,
        f"--policy.auto_patch_k={cfg.k}" ,
        f"--policy.auto_patch_alpha={cfg.alpha}" ,
        f"--policy.auto_patch_top_k_tokens=5" ,
        f"--env.type=libero" ,
        f"--env.task=libero_object" ,
        f"--env.control_mode=absolute",
        f"--eval.batch_size=2" ,
        f"--eval.n_episodes=5" ,
        f"--eval.condition_label=xvla-{cfg.target}-{cfg.k}-{cfg.alpha}" ,
        f"--output_dir=./logs/xvla-{cfg.target}-{cfg.k}-{cfg.alpha}"
    ]

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run()

lerobot-train   --policy.path=lerobot/smolvla_base   --dataset.repo_id=lerobot/droid_100   --batch_size=64   --steps=10   --output_dir=lerobot/src/lerobot/outputs/train/my_smolvla_droid_hf   --job_name=my_smolvla_training   --policy.device=cuda   --wandb.enable=true --policy.repo_id=adalal/smolvla_test --rename_map='{"observation.images.exterior_image_1_left": "observation.images.camera1", "observation.images.exterior_image_2_left": "observation.images.camera2", "observation.images.wrist_image_left": "observation.images.camera3"}'

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/svla_so100_stacking \
  --batch_size=64 \
  --steps=20000  \
  --output_dir=lerobot/src/lerobot/outputs/train/smolvla_so100_stacking \
  --job_name=my_smolvla_training   --policy.device=cuda   --wandb.enable=true --policy.repo_id=adalal/smolvla_stacking_test \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

lerobot-train \
  --policy.type=smolvla \
  --policy.repo_id=adalal/smolvla-libero-64 \
  --policy.load_vlm_weights=true \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --output_dir=./outputs/smolvla-libero-64 \
  --steps=100000 \
  --batch_size=64 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval_freq=1000 \
  --wandb.enable=true


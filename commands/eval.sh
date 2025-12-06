# SMOL VLA
lerobot-eval \
  --policy.path="adalal/smolvla-libero-64-2.0" \
  --policy.n_action_steps=10 \
  --policy.auto_patch_random=false \
  --policy.auto_patch_target="TARGET_WORD" \
  --policy.auto_patch_k=NUM_VECTORS_TO_PATCH \
  --policy.auto_patch_alpha=PATCH_STRENGTH \
  --policy.auto_patch_top_k_tokens=5 \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=5 \
  --eval.n_episodes=5 \
  --eval.condition_label=RUN_ID \
  --output_dir=./logs/RUN_ID

# X-VLA
lerobot-eval \
  --policy.path="lerobot/xvla-libero" \
  --policy.n_action_steps=10 \
  --policy.auto_patch_random=false \
  --policy.auto_patch_target="TARGET_WORD" \
  --policy.auto_patch_k=NUM_VECTORS_TO_PATCH \
  --policy.auto_patch_alpha=PATCH_STRENGTH \
  --policy.auto_patch_top_k_tokens=5 \
  --env.type=libero \
  --env.task=libero_object \
  --env.control_mode=absolute \
  --eval.batch_size=5 \
  --eval.n_episodes=5 \
  --eval.condition_label=RUN_ID \
  --output_dir=./logs/RUN_ID

  lerobot-eval \
  --policy.path="adalal/smolvla-libero-64-2.0" \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=5 \
  --eval.n_episodes=5 \
  --eval.condition_label=smolvla-down-prepend \
  --output_dir=./logs/smolvla-down-prepend
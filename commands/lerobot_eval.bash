lerobot-eval \
    --policy.path="adalal/smolvla-libero-object" \
    --env.type=libero \
    --env.task=libero_object \
    --eval.batch_size=2 \
    --eval.n_episodes=3 \
    --output_dir=../eval_outputs/smolvla_libero_object \
    --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
    --policy.output_features='{"action": {"type": "ACTION", "shape": [7]}}'

lerobot-eval \
    --policy.path="adalal/smolvla-libero-object" \
    --env.type=libero \
    --env.task=libero_object \
    --eval.batch_size=2 \
    --eval.n_episodes=3 \
    --output_dir=../eval_outputs/smolvla_libero_object

lerobot-eval \
    --policy.path="./outputs/smolvla-libero-64-2/checkpoints/last/pretrained_model/" \
    --policy.n_action_steps=10 \
    --env.type=libero \
    --env.task=libero_object \
    --eval.batch_size=2 \
    --eval.n_episodes=3 \
    --output_dir=../eval_outputs/smolvla_libero_object_64_50k

python lerobot_eval_perturbation.py \
    --policy.path="lerobot/smolvla_base" \
    --env.type=libero \
    --env.task=libero_object \
    --eval.batch_size=2 \
    --eval.n_episodes=3 \
    --output_dir=../eval_outputs/smolvla_base_per \
    --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
    --policy.output_features='{"action": {"type": "ACTION", "shape": [7]}}'


python lerobot_eval_perturbation.py \
    --policy.path="adalal/smolvla-libero-object" \
    --env.type=libero \
    --env.task=libero_object \
    --eval.batch_size=2 \
    --eval.n_episodes=3 \
    --output_dir=../eval_outputs/smolvla_libero_object \
    --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
    --policy.output_features='{"action": {"type": "ACTION", "shape": [7]}}'
#!/usr/bin/env python

# ------------------------------------------------------------------------------
# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations

import builtins
import logging
import os
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE

from .action_hub import build_action_space
from .configuration_florence2 import Florence2Config
from .configuration_xvla import XVLAConfig
from .modeling_florence2 import Florence2ForConditionalGeneration
from .soft_transformer import SoftPromptedTransformer


def _create_ffn_patch_hook(vector_indices, alpha):
    """Create a hook for down_proj that patches intermediate activations."""

    def hook(module, input):
        modified = input[0].clone()
        for idx in vector_indices:
            modified[..., idx] = alpha
        return (modified,)

    return hook


def extract_vlm_value_vectors(policy: "XVLAPolicy"):
    """Extract all value vectors from Florence2 encoder FFN layers."""
    layers = policy.model.vlm.language_model.model.encoder.layers
    value_vectors = []
    for layer_idx, layer in enumerate(layers):
        down_proj_weight = layer.fc2.weight.data
        for vector_idx in range(down_proj_weight.shape[1]):
            value_vector = down_proj_weight[:, vector_idx]
            value_vectors.append((layer_idx, vector_idx, value_vector))
    return value_vectors


def compute_semantic_embeddings(
    value_vectors,
    embeddings,
    final_bias=None,
    layer_norms=None,
    top_k=5,
    chunk_size=512,
):
    """
    Convert value vectors to semantic embeddings via unembedding.

    If `final_bias` is provided, it is added to logits (to mirror Florence2 forward which applies
    `final_logits_bias`). Chunking prevents building an enormous (num_vectors x vocab) logits tensor.
    """

    device = embeddings.device
    semantic_embs = []
    all_top_indices = []
    all_top_probs = []

    with torch.no_grad():
        for start in range(0, len(value_vectors), chunk_size):
            chunk = value_vectors[start : start + chunk_size]
            raw_vectors = torch.stack([v for _, _, v in chunk]).to(device)
            raw_norm = raw_vectors.norm(dim=1)
            vectors = raw_vectors

            if layer_norms is not None:
                ln_states = []
                for (layer_idx, _, _), vec in zip(chunk, raw_vectors):
                    weight, bias, eps = layer_norms[layer_idx]
                    weight = weight.to(device)
                    bias = bias.to(device)
                    mean = vec.mean()
                    var = vec.var(unbiased=False)
                    normalized = (vec - mean) / torch.sqrt(var + eps)
                    ln_states.append(weight * normalized + bias)
                vectors = torch.stack(ln_states, dim=0)
            vec_norm = vectors.norm(dim=1)

            logits = F.linear(vectors, embeddings)
            if final_bias is not None:
                logits = logits + final_bias.to(device)

            top_logits, top_indices = torch.topk(logits, k=top_k, dim=1)
            top_probs = F.softmax(top_logits, dim=1)
            top_gap = (top_logits[:, 0] - top_logits[:, -1]).detach()

            token_embs = embeddings[top_indices]
            semantic_chunk = (top_probs.unsqueeze(-1) * token_embs).sum(dim=1)

            semantic_embs.append(semantic_chunk)
            all_top_indices.append(top_indices)
            all_top_probs.append(top_probs)

            del vectors, logits, top_logits, top_indices, top_probs, token_embs, semantic_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    semantic_embs = torch.cat(semantic_embs, dim=0)
    all_top_indices = torch.cat(all_top_indices, dim=0)
    all_top_probs = torch.cat(all_top_probs, dim=0)

    return semantic_embs, all_top_indices, all_top_probs


def find_nearest_neighbors(target_word, semantic_embeddings, value_vectors, embeddings, tokenizer, k=20):
    """Find k value vectors with semantic embeddings closest to target word."""
    target_token_id = tokenizer.encode(target_word, add_special_tokens=False)[0]
    target_emb = embeddings[target_token_id]
    semantic_norm = F.normalize(semantic_embeddings, dim=1)
    target_norm = F.normalize(target_emb.unsqueeze(0), dim=1)
    similarities = (semantic_norm @ target_norm.T).squeeze()
    top_sims, top_indices = torch.topk(similarities, k=min(k, len(similarities)))

    results = []
    for sim, idx in zip(top_sims, top_indices):
        layer_idx, vector_idx, _ = value_vectors[idx]
        results.append((layer_idx, vector_idx, sim.item()))
    return results


def generate_patches(neighbors, alpha):
    """Group neighbors by layer and generate patch config."""
    patches = {}
    for layer_idx, vector_idx, _ in neighbors:
        patches.setdefault(layer_idx, []).append(vector_idx)
    return [
        {"layer_idx": layer_idx, "vector_indices": indices, "alpha": alpha}
        for layer_idx, indices in sorted(patches.items())
    ]


class XVLAModel(nn.Module):
    """
    XVLA backbone that stitches Florence-2 embeddings with the temporal/action transformer head.
    """

    def __init__(
        self,
        config: XVLAConfig,
        florence_config: Florence2Config,
        proprio_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.chunk_size: int = config.chunk_size
        self.use_proprio: bool = config.use_proprio

        # Build action space with auto-detection for "auto" mode
        if config.action_mode.lower() == "auto":
            # Auto-detect real action dim from config.action_feature
            real_dim = (
                config.action_feature.shape[-1]
                if config.action_feature is not None
                else config.max_action_dim
            )
            self.action_space = build_action_space(
                config.action_mode.lower(),
                real_dim=real_dim,
                max_dim=config.max_action_dim,
            )
        else:
            self.action_space = build_action_space(config.action_mode.lower())

        self.dim_action = self.action_space.dim_action
        self.dim_proprio = proprio_dim

        self.vlm = Florence2ForConditionalGeneration(florence_config)
        if hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                del lm.model.decoder
            # Remove lm_head; activation patching uses encoder embeddings directly.
            if hasattr(lm, "lm_head"):
                del lm.lm_head

        projection_dim = getattr(self.vlm.config, "projection_dim", None)
        if projection_dim is None:
            raise ValueError("Florence2 config must provide `projection_dim` for multimodal fusion.")

        self.transformer = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=self.dim_action,
            dim_propio=self.dim_proprio,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
        )

        # Apply freezing based on config
        self._apply_freezing()

        # Apply dtype casting based on config
        self._apply_dtype()

    def _get_target_dtype(self) -> torch.dtype:
        """Get the target dtype based on config."""
        if self.config.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def _apply_dtype(self) -> None:
        """
        Apply dtype casting to model components based on config.
        """
        target_dtype = self._get_target_dtype()
        self.to(dtype=target_dtype)

    def _apply_freezing(self) -> None:
        """
        Freeze VLM vision and language encoders based on config options.
        Keep only policy transformer and soft prompts trainable.
        """
        # Freeze vision encoder
        if self.config.freeze_vision_encoder and hasattr(self.vlm, "vision_tower"):
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = False

        # Freeze language encoder
        if self.config.freeze_language_encoder and hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            # Freeze encoder
            if hasattr(lm, "model") and hasattr(lm.model, "encoder"):
                for param in lm.model.encoder.parameters():
                    param.requires_grad = False
            # Freeze shared embeddings
            if hasattr(lm, "model") and hasattr(lm.model, "shared"):
                for param in lm.model.shared.parameters():
                    param.requires_grad = False

        # Freeze or unfreeze policy transformer
        if not self.config.train_policy_transformer:
            for name, param in self.transformer.named_parameters():
                if "soft_prompts" not in name:
                    param.requires_grad = False

        # Freeze or unfreeze soft prompts
        if not self.config.train_soft_prompts and hasattr(self.transformer, "soft_prompt_hub"):
            for param in self.transformer.soft_prompt_hub.parameters():
                param.requires_grad = False

    def forward_vlm(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Encode text and multi-view images via Florence2 encoder.
        """
        batch_size, num_views = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(dtype=torch.bool)
        flat_images = pixel_values.flatten(0, 1)
        num_valid = int(flat_mask.sum().item())
        if num_valid == 0:
            raise ValueError("At least one image view must be valid per batch.")

        valid_images = flat_images[flat_mask]
        valid_feats = self.vlm._encode_image(valid_images)
        tokens_per_view, hidden_dim = valid_feats.shape[1:]

        image_features = valid_feats.new_zeros((batch_size * num_views, tokens_per_view, hidden_dim))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(batch_size, num_views, tokens_per_view, hidden_dim)
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)
        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],
            inputs_embeds,
        )

        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]

        aux_visual_inputs = image_features[:, 1:].reshape(batch_size, -1, hidden_dim)
        return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}

    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for the XVLA model.
        """
        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)
        action = action.to(dtype=target_dtype)

        enc = self.forward_vlm(input_ids, image_input, image_mask)

        batch_size = input_ids.shape[0]
        t = (
            torch.rand(1, device=input_ids.device, dtype=target_dtype)
            + torch.arange(batch_size, device=input_ids.device, dtype=target_dtype) / batch_size
        ) % (1 - 1e-5)

        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        pred_action = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            **enc,
        )
        return self.action_space.compute_loss(pred_action, action)

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        self.eval()

        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)

        enc = self.forward_vlm(input_ids, image_input, image_mask)

        batch_size = input_ids.shape[0]
        action_dim = self.dim_action

        x1 = torch.randn(batch_size, self.chunk_size, action_dim, device=proprio.device, dtype=target_dtype)
        action = torch.zeros_like(x1)

        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((batch_size,), i / steps, device=proprio.device, dtype=target_dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )
        return self.action_space.postprocess(action)


class XVLAPolicy(PreTrainedPolicy):
    """LeRobot-compliant wrapper built around the XVLA model."""

    config_class = XVLAConfig
    name = "xvla"

    def __init__(self, config: XVLAConfig):
        super().__init__(config)
        config.validate_features()
        florence_config = config.get_florence_config()
        proprio_dim = config.max_state_dim if config.use_proprio else 0
        self.model = XVLAModel(config=config, florence_config=florence_config, proprio_dim=proprio_dim)
        self.reset()

        self._activation_patches = None

    def reset(self) -> None:
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        self._patch_hooks = []

    def _compute_auto_patches(self):
        """Compute activation patches automatically based on semantic value vectors."""
        config = self.config
        vlm = self.model.vlm.language_model

        # lm_head is stripped in XVLA; use the tied encoder embeddings directly for unembedding.
        embeddings = vlm.model.encoder.embed_tokens.weight.data.to(config.device)
        final_bias = getattr(vlm, "final_logits_bias", None)
        print(
            f"Auto-patch projection uses final_logits_bias={'yes' if final_bias is not None else 'no'}."
        )
        layer_norms = {
            idx: (
                layer.final_layer_norm.weight.data.clone(),
                layer.final_layer_norm.bias.data.clone(),
                layer.final_layer_norm.eps,
            )
            for idx, layer in enumerate(vlm.model.encoder.layers)
        }
        tokenizer = self._get_tokenizer()

        print("Extracting value vectors...")
        value_vectors = extract_vlm_value_vectors(self)
        print(f"Found {len(value_vectors)} value vectors across {len(set(v[0] for v in value_vectors))} layers")

        print(f"Computing semantic embeddings (top-{config.auto_patch_top_k_tokens} tokens)...")
        semantic_embeddings, top_tokens, top_probs = compute_semantic_embeddings(
            value_vectors,
            embeddings,
            final_bias,
            layer_norms,
            top_k=config.auto_patch_top_k_tokens,
            chunk_size=512,
        )

        print(f"Finding k={config.auto_patch_k} nearest neighbors to '{config.auto_patch_target}'...")
        neighbors = find_nearest_neighbors(
            config.auto_patch_target,
            semantic_embeddings,
            value_vectors,
            embeddings,
            tokenizer,
            k=config.auto_patch_k,
        )

        print(f"\nTop value vectors for '{config.auto_patch_target}':")
        for i, (layer_idx, vector_idx, sim) in enumerate(neighbors[:10]):
            print(f"  {i+1}. Layer {layer_idx}, Vector {vector_idx}: similarity={sim:.4f}")
            vec_idx = next(j for j, (l, v, _) in enumerate(value_vectors) if l == layer_idx and v == vector_idx)
            tokens = top_tokens[vec_idx]
            probs = top_probs[vec_idx]
            print("      Top tokens: ", end="")
            for tok_id, prob in zip(tokens, probs):
                tok_str = tokenizer.decode([tok_id.item()])
                print(f"'{tok_str}'({prob:.2f}) ", end="")
            print()

        vlm_patches = generate_patches(neighbors, config.auto_patch_alpha)
        print(f"\nGenerated {len(vlm_patches)} VLM layer patches with alpha={config.auto_patch_alpha}")
        return {
            "vlm": vlm_patches,
            "transformer": [],
        }

    def _register_activation_patches(self):
        """Register FFN activation patches on Florence2 encoder fc2 layers."""
        vlm_layers = self.model.vlm.language_model.model.encoder.layers

        for patch in self._activation_patches.get("vlm", []):
            hook = vlm_layers[patch["layer_idx"]].fc2.register_forward_pre_hook(
                _create_ffn_patch_hook(patch["vector_indices"], patch["alpha"])
            )
            self._patch_hooks.append(hook)

        # Placeholder for transformer patches if provided manually
        if "transformer" in self._activation_patches:
            transformer_layers = self.model.transformer.blocks
            for patch in self._activation_patches.get("transformer", []):
                hook = transformer_layers[patch["layer_idx"]].mlp.fc2.register_forward_pre_hook(
                    _create_ffn_patch_hook(patch["vector_indices"], patch["alpha"])
                )
                self._patch_hooks.append(hook)

    def _get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.config.tokenizer_name, padding_side=self.config.tokenizer_padding_side)

    def get_optim_params(self) -> dict:
        """Return trainable named parameters for optimization.

        Returns a dict of name -> param for all trainable parameters.
        This enables the xvla-adamw optimizer to apply differential learning rates
        based on parameter names (e.g., 1/10 LR for VLM components).
        """
        return dict(filter(lambda kv: kv[1].requires_grad, self.named_parameters()))

    def _prepare_state(self, batch: dict[str, Tensor], batch_size: int, device: torch.device) -> Tensor:
        if not self.config.use_proprio or OBS_STATE not in batch:
            return torch.zeros(batch_size, 0, device=device)
        state = batch[OBS_STATE]
        if state.ndim > 2:
            state = state[:, -1, :]
        return pad_vector(state, self.model.dim_proprio)

    def _prepare_images(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        present_img_keys = [key for key in self.config.image_features if key in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                "All image features are missing from the batch. "
                f"Batch keys: {list(batch.keys())}, expected at least one of {list(self.config.image_features)}."
            )

        images = []
        masks = []
        for key in present_img_keys:
            img = batch[key][:, -1] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding)
            images.append(img)
            masks.append(torch.ones(img.size(0), dtype=torch.bool, device=img.device))

        stacked_imgs = torch.stack(images, dim=1)
        stacked_masks = torch.stack(masks, dim=1)

        total_views = self.config.num_image_views or stacked_imgs.size(1)
        total_views = max(total_views, stacked_imgs.size(1))
        num_pad = total_views - stacked_imgs.size(1)
        if num_pad > 0:
            pad_shape = (stacked_imgs.size(0), num_pad, *stacked_imgs.shape[2:])
            pad_imgs = stacked_imgs.new_zeros(pad_shape)
            pad_masks = stacked_masks.new_zeros((stacked_masks.size(0), num_pad))
            stacked_imgs = torch.cat([stacked_imgs, pad_imgs], dim=1)
            stacked_masks = torch.cat([stacked_masks, pad_masks], dim=1)

        return stacked_imgs, stacked_masks

    def _get_domain_id(self, batch: dict[str, Tensor], batch_size: int, device: torch.device) -> Tensor:
        candidate = None
        if self.config.domain_feature_key and self.config.domain_feature_key in batch:
            candidate = batch[self.config.domain_feature_key]
        elif "domain_id" in batch:
            candidate = batch["domain_id"]

        if candidate is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        if not isinstance(candidate, torch.Tensor):
            candidate = torch.as_tensor(candidate, device=device)
        else:
            candidate = candidate.to(device=device)

        if candidate.ndim == 0:
            candidate = candidate.expand(batch_size)
        if candidate.ndim > 1:
            candidate = candidate.view(candidate.shape[0], -1)[:, 0]
        if candidate.shape[0] != batch_size:
            candidate = candidate.expand(batch_size)
        return candidate.to(dtype=torch.long)

    def _prepare_action_targets(self, batch: dict[str, Tensor]) -> Tensor:
        if ACTION not in batch:
            raise ValueError("Batch is missing action targets required for training.")
        actions = batch[ACTION]
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        actions = pad_tensor_along_dim(actions, self.config.chunk_size, dim=1)
        if actions.shape[-1] != self.model.dim_action:
            actions = pad_vector(actions, self.model.dim_action)
        return actions

    def _build_model_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        input_ids = batch[OBS_LANGUAGE_TOKENS]
        batch_size = input_ids.shape[0]
        images, image_mask = self._prepare_images(batch)
        domain_id = self._get_domain_id(batch, batch_size, images.device)
        proprio = self._prepare_state(batch, batch_size, images.device)
        return {
            "input_ids": input_ids,
            "image_input": images,
            "image_mask": image_mask,
            "domain_id": domain_id,
            "proprio": proprio,
        }

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        inputs = self._build_model_inputs(batch)
        targets = self._prepare_action_targets(batch)
        losses = self.model(action=targets, **inputs)
        total_loss = sum(losses.values())

        log_dict = {k: v.detach().item() for k, v in losses.items()}
        log_dict["loss"] = total_loss.detach().item()
        return total_loss, log_dict

    def _get_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        inputs = self._build_model_inputs(batch)
        actions = self.model.generate_actions(**inputs, steps=self.config.num_denoising_steps)
        return actions

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:  # noqa: ARG002
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        return self._get_action_chunk(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:  # noqa: ARG002
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ):
        """
        Loads XVLA model weights with:
        - automatic prefix 'model.' added to all keys
        - skip list for layers that should remain randomly initialized
        """
        import safetensors.torch

        # step 1: load config
        # TODO: jadechoghari, fix this
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        # step 2: locate model.safetensors
        if os.path.isdir(model_id):
            logging.info("Loading weights from local directory")
            model_file = os.path.join(model_id, "model.safetensors")
        else:
            try:
                from huggingface_hub import hf_hub_download
                from huggingface_hub.utils import HfHubHTTPError

                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename="model.safetensors",
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(f"model.safetensors not found on the Hub at {model_id}") from e

        logging.info(f"Loading checkpoint from {model_file}")
        # step 3: load state dict
        state_dict = safetensors.torch.load_file(model_file)
        encoder_key = "model.vlm.language_model.model.encoder.embed_tokens.weight"
        shared_key = "model.vlm.language_model.model.shared.weight"
        if encoder_key in state_dict:
            state_dict[shared_key] = state_dict[encoder_key]
            # or deepcopy
        # step 4: load into instance
        instance.load_state_dict(state_dict, strict=True)
        logging.info("Loaded XVLA checkpoint")
        # step 5: finalize
        # Reapply dtype after loading state dict
        instance.model._apply_dtype()
        instance.to(config.device)
        instance.eval()
        # Compute activation patches now that weights are loaded
        if instance.config.auto_patch_target is not None:
            # Remove any existing hooks before re-registering
            for hook in getattr(instance, "_patch_hooks", []):
                hook.remove()
            instance._patch_hooks = []
            instance._activation_patches = instance._compute_auto_patches()
            instance._register_activation_patches()
        return instance


def resize_with_pad(img: torch.Tensor, height: int, width: int, pad_value: float = 0.0) -> torch.Tensor:
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but got {img.shape}")

    current_height, current_width = img.shape[2:]
    if current_height == height and current_width == width:
        return img

    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, height - resized_height)
    pad_width = max(0, width - resized_width)
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    if vector.shape[-1] == new_dim:
        return vector
    if new_dim == 0:
        shape = list(vector.shape)
        shape[-1] = 0
        return vector.new_zeros(*shape)
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = vector.new_zeros(*shape)
    length = min(current_dim, new_dim)
    new_vector[..., :length] = vector[..., :length]
    return new_vector


def pad_tensor_along_dim(tensor: Tensor, target_len: int, dim: int = 1) -> Tensor:
    current_len = tensor.size(dim)
    if current_len == target_len:
        return tensor
    if current_len > target_len:
        slices = [slice(None)] * tensor.dim()
        slices[dim] = slice(0, target_len)
        return tensor[tuple(slices)]
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_len - current_len
    pad_tensor = tensor.new_zeros(pad_shape)
    return torch.cat([tensor, pad_tensor], dim=dim)

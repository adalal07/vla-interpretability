#!/usr/bin/env python3
"""
Find FFN value vectors semantically related to a target word.

Approach:
1. Extract value vectors (columns of down_proj.weight) from VLM FFN layers
2. Project each through unembedding: final_norm → lm_head → logits
3. Create semantic embedding: softmax-weighted average of top-k token embeddings
4. Find k-NN to target word embedding
5. Output patches for ACTIVATION_PATCHES
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def extract_vlm_value_vectors(model):
    """Extract all value vectors from VLM FFN layers.

    Returns: list of (layer_idx, vector_idx, value_vector_tensor)
    """
    layers = model.model.vlm_with_expert.get_vlm_model().text_model.layers
    value_vectors = []

    for layer_idx, layer in enumerate(layers):
        # down_proj.weight: [hidden_size, intermediate_size]
        # Each column is a value vector
        down_proj_weight = layer.mlp.down_proj.weight.data

        for vector_idx in range(down_proj_weight.shape[1]):
            value_vector = down_proj_weight[:, vector_idx]
            value_vectors.append((layer_idx, vector_idx, value_vector))

    return value_vectors


def compute_semantic_embeddings(
    value_vectors, embeddings, lm_head, final_norm, top_k=5
):
    """Convert value vectors to semantic embeddings via unembedding.

    For each value vector v:
      1. logits = lm_head(final_norm(v))
      2. top_tokens = topk(logits, k)
      3. semantic_emb = sum_i softmax(logits[i]) * embeddings[i]

    Returns:
        semantic_embeddings: tensor of shape [num_vectors, hidden_dim]
        top_tokens: tensor of shape [num_vectors, top_k] - token IDs
        top_probs: tensor of shape [num_vectors, top_k] - probabilities
    """
    vectors = torch.stack([v for _, _, v in value_vectors])

    # Project through unembedding
    normed = final_norm(vectors)
    logits = lm_head(normed)

    # Get top-k tokens and their probabilities
    top_logits, top_indices = torch.topk(logits, k=top_k, dim=1)
    top_probs = F.softmax(top_logits, dim=1)

    # Semantic embedding = weighted average of top token embeddings
    semantic_embs = []
    for i in range(len(vectors)):
        token_embs = embeddings[top_indices[i]]  # [top_k, hidden_dim]
        weights = top_probs[i]  # [top_k]
        semantic_emb = (weights[:, None] * token_embs).sum(dim=0)
        semantic_embs.append(semantic_emb)

    return torch.stack(semantic_embs), top_indices, top_probs


def find_nearest_neighbors(
    target_word, semantic_embeddings, value_vectors, embeddings, tokenizer, k=20
):
    """Find k value vectors with semantic embeddings closest to target word.

    Returns: list of (layer_idx, vector_idx, similarity)
    """
    # Get target embedding
    target_token_id = tokenizer.encode(target_word, add_special_tokens=False)[0]
    target_emb = embeddings[target_token_id]

    # Cosine similarity
    semantic_norm = F.normalize(semantic_embeddings, dim=1)
    target_norm = F.normalize(target_emb.unsqueeze(0), dim=1)
    similarities = (semantic_norm @ target_norm.T).squeeze()

    # Top-k
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
        if layer_idx not in patches:
            patches[layer_idx] = []
        patches[layer_idx].append(vector_idx)

    patch_list = [
        {"layer_idx": layer_idx, "vector_indices": indices, "alpha": alpha}
        for layer_idx, indices in sorted(patches.items())
    ]

    return patch_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lerobot/smolvla_base")
    parser.add_argument("--target", default="slow")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--top-k-tokens", type=int, default=5)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    # Load model
    print(f"Loading {args.model}...")
    model = SmolVLAPolicy.from_pretrained(args.model).to(args.device).eval()

    # Get components
    vlm = model.model.vlm_with_expert.vlm
    vlm_text = model.model.vlm_with_expert.get_vlm_model().text_model
    embeddings = vlm_text.embed_tokens.weight.data.to(args.device)
    lm_head = vlm.lm_head
    final_norm = vlm_text.norm
    tokenizer = model.model.vlm_with_expert.processor.tokenizer

    print(f"Extracting value vectors...")
    value_vectors = extract_vlm_value_vectors(model)
    print(
        f"Found {len(value_vectors)} value vectors across {len(set(v[0] for v in value_vectors))} layers"
    )

    print(f"Computing semantic embeddings (top-{args.top_k_tokens} tokens)...")
    semantic_embeddings, top_tokens, top_probs = compute_semantic_embeddings(
        value_vectors, embeddings, lm_head, final_norm, top_k=args.top_k_tokens
    )

    print(f"Finding k={args.k} nearest neighbors to '{args.target}'...")
    neighbors = find_nearest_neighbors(
        args.target, semantic_embeddings, value_vectors, embeddings, tokenizer, k=args.k
    )

    # Print results
    print(f"\nTop value vectors for '{args.target}':")
    for i, (layer_idx, vector_idx, sim) in enumerate(neighbors[:10]):
        print(f"  {i+1}. Layer {layer_idx}, Vector {vector_idx}: similarity={sim:.4f}")

        # Find this value vector's index to get its top tokens
        vec_idx = next(
            j
            for j, (l, v, _) in enumerate(value_vectors)
            if l == layer_idx and v == vector_idx
        )
        tokens = top_tokens[vec_idx]
        probs = top_probs[vec_idx]

        print(f"      Top tokens: ", end="")
        for tok_id, prob in zip(tokens, probs):
            tok_str = tokenizer.decode([tok_id.item()])
            print(f"'{tok_str}'({prob:.2f}) ", end="")
        print()

    # Generate patches
    patches = generate_patches(neighbors, args.alpha)

    print(f"\n{'='*80}")
    print("ACTIVATION_PATCHES = {")
    print('    "vlm": [')
    for patch in patches:
        print(f"        {patch},")
    print("    ],")
    print('    "expert": [],')
    print("}")
    print("=" * 80)


if __name__ == "__main__":
    main()

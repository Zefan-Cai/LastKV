"""Exact inference cache: shared deep history, local first-frame and recent KV."""

import torch


def allocate_caches(model, batch_size, total_tokens, block_tokens, dtype, device,
                    shared=True):
    if total_tokens < 0 or block_tokens <= 0:
        raise ValueError("Invalid history or block capacity.")
    if shared and model.v_layer_idxs != [model.num_layers - 1] * model.num_layers:
        raise ValueError("Shared history requires every layer to reuse the final layer KV.")
    shape = (batch_size, total_tokens, model.num_heads, model.dim // model.num_heads)
    deep_k = torch.empty(shape, dtype=dtype, device=device) if shared else None
    deep_v = torch.empty_like(deep_k) if shared else None
    prefix_tokens = min(total_tokens, model.force_first_local_kv_tokens)
    recent_tokens = min(total_tokens, block_tokens)
    caches = []
    for _ in range(model.num_layers):
        k = deep_k if shared else torch.empty(shape, dtype=dtype, device=device)
        v = deep_v if shared else torch.empty_like(k)
        recent_shape = (batch_size, recent_tokens, model.num_heads, model.dim // model.num_heads)
        ck = torch.empty(recent_shape, dtype=dtype, device=device)
        cv = torch.empty_like(ck)
        cache = {"k_storage": k, "v_storage": v, "k": k[:, :0], "v": v[:, :0],
                 "cache_len": torch.zeros(1, dtype=torch.long, device=device),
                 "curr_k_storage": ck, "curr_v_storage": cv,
                 "curr_k": ck[:, :0], "curr_v": cv[:, :0], "curr_start": 0,
                 "curr_cache_max_len": recent_tokens, "prev_block_alpha": None,
                 "force_first_local_kv_tokens": model.force_first_local_kv_tokens,
                 "shared_deep": shared}
        if shared:
            prefix_shape = (batch_size, prefix_tokens, model.num_heads, model.dim // model.num_heads)
            cache["prefix_k"] = torch.empty(prefix_shape, dtype=dtype, device=device)
            cache["prefix_v"] = torch.empty_like(cache["prefix_k"])
        caches.append(cache)
    return caches


def append_shared(cache, k, v, start):
    if torch.is_grad_enabled():
        raise RuntimeError("Shared inference caches require torch.no_grad().")
    if start is None:
        start = cache["cache_len"].item()
    end = start + k.shape[1]
    if start != cache["cache_len"].item() or end > cache["k_storage"].shape[1]:
        raise ValueError("Noncontiguous cache append or exceeded cache capacity.")
    # All layers select the same final-layer KV outside their retained prefix.
    # Every clean block finishes all layer forwards before this commit loop.
    cache["k_storage"][:, start:end].copy_(k)
    cache["v_storage"][:, start:end].copy_(v)
    prefix_end = min(end, cache["prefix_k"].shape[1])
    if start < prefix_end:
        cache["prefix_k"][:, start:prefix_end].copy_(k[:, :prefix_end - start])
        cache["prefix_v"][:, start:prefix_end].copy_(v[:, :prefix_end - start])
    cache["cache_len"].fill_(end)


def history_kv(cache, length):
    if length > cache["cache_len"].item():
        raise ValueError("Requested history exceeds committed KV.")
    retained = min(length, cache["prefix_k"].shape[1])
    if retained == length:
        return cache["prefix_k"][:, :length], cache["prefix_v"][:, :length]
    return (torch.cat((cache["prefix_k"][:, :retained], cache["k_storage"][:, retained:length]), dim=1),
            torch.cat((cache["prefix_v"][:, :retained], cache["v_storage"][:, retained:length]), dim=1))

"""Prefill/decode disaggregation hand-off cost: bytes and time to move one
request's state from a prefill GPU to a decode GPU.

LastKV state for a request with T prompt tokens = shared final-layer K/V
(``T * 2 * Hkv * D * 2`` bytes, 4 KiB/token for the 8B model) + the per-layer
two-chunk ring (``L * 2C * 2 * Hkv * D * 2`` bytes, 256 MiB). A Transformer
hands off its whole KV cache (``T * L * 2 * Hkv * D * 2``, 128 KiB/token).
The copy is a plain device-to-device ``copy_`` between two visible GPUs
(peer access if available, else staged through the host by the driver).

    CUDA_VISIBLE_DEVICES=3,6 python tools/pd_transfer_bench.py --lens 24576,32768,65536,126976
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch


def time_copy(src: torch.Tensor, dst: torch.Tensor, reps: int) -> float:
    dst.copy_(src)
    torch.cuda.synchronize(src.device)
    torch.cuda.synchronize(dst.device)
    times = []
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        dst.copy_(src, non_blocking=True)
        e1.record()
        e1.synchronize()
        times.append(e0.elapsed_time(e1))
    return statistics.median(times)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lens", default="24576,32768,65536,126976")
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--out")
    args = p.parse_args()
    if torch.cuda.device_count() < 2:
        raise SystemExit("need two visible GPUs")
    a, b = torch.device("cuda:0"), torch.device("cuda:1")
    peer = torch.cuda.can_device_access_peer(0, 1)
    per_tok_layer = 2 * args.kv_heads * args.head_dim * 2
    ring_bytes = args.layers * 2 * args.chunk * per_tok_layer
    rows = []
    for T in (int(x) for x in args.lens.split(",")):
        lk_bytes = T * per_tok_layer + ring_bytes
        tf_bytes = T * args.layers * per_tok_layer
        lk_src = torch.empty(lk_bytes // 2, dtype=torch.bfloat16, device=a)
        lk_dst = torch.empty_like(lk_src, device=b)
        ms_lk = time_copy(lk_src, lk_dst, args.reps)
        del lk_src, lk_dst
        try:
            tf_src = torch.empty(tf_bytes // 2, dtype=torch.bfloat16, device=a)
            tf_dst = torch.empty_like(tf_src, device=b)
            ms_tf = time_copy(tf_src, tf_dst, args.reps)
            del tf_src, tf_dst
        except torch.cuda.OutOfMemoryError:
            ms_tf = None
        torch.cuda.empty_cache()
        row = {
            "tokens": T,
            "lastkv_gb": round(lk_bytes / 1e9, 3),
            "transformer_gb": round(tf_bytes / 1e9, 3),
            "bytes_ratio": round(tf_bytes / lk_bytes, 1),
            "lastkv_ms": round(ms_lk, 2),
            "transformer_ms": round(ms_tf, 2) if ms_tf is not None else "oom",
            "lastkv_gbps": round(lk_bytes / 1e9 / (ms_lk / 1e3), 1),
            "transformer_gbps": round(tf_bytes / 1e9 / (ms_tf / 1e3), 1) if ms_tf is not None else None,
        }
        print(row)
        rows.append(row)
    summary = {"script": "pd_transfer_bench", "gpu": torch.cuda.get_device_name(0), "peer_access": peer,
               "layers": args.layers, "kv_heads": args.kv_heads, "head_dim": args.head_dim, "chunk": args.chunk, "rows": rows}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

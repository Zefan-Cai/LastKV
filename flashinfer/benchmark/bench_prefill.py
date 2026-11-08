"""Prefill throughput vs prompt length for the standalone FlashInfer engine.

One engine is built for the longest length; each length gets one warmup and
``--trials`` timed ``generate()`` calls. Decode is ``--new`` tokens (default 1)
so the reported prefill time dominates. Prints a per-trial table and exactly
one machine-readable JSON object as the last stdout line.

    python benchmark/bench_prefill.py --model /ckpt --prompt-json p.json \
        --lens 8192,32768,126976 --batch 4 --out prefill.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GIB = 2 ** 30
SEED = 1234
DEFAULT_LENS = "8192,16384,32768,65536,126976"


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def read_vocab_size(model_path: str) -> int:
    with open(os.path.join(model_path, "config.json")) as f:
        return int(json.load(f)["vocab_size"])


def base_token_ids(args: argparse.Namespace, max_len: int, vocab: int) -> list[int]:
    """Token ids used (sliced) for every measured length."""
    if args.prompt_json:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
        with open(args.prompt_json) as f:
            text = json.load(f)["prompt"]
        ids = tok.encode(text, add_special_tokens=False).ids
        if len(ids) < max_len:
            raise SystemExit(
                f"--prompt-json tokenises to {len(ids)} tokens, need {max_len}"
            )
        return ids
    rng = random.Random(SEED)
    return [rng.randrange(vocab) for _ in range(max_len)]


def build_prompts(ids: list[int], length: int, batch: int, vocab: int) -> list[list[int]]:
    """``batch`` copies of the first ``length`` ids, unique first token per request."""
    prompts = []
    for i in range(batch):
        p = list(ids[:length])
        p[0] = (p[0] + i) % vocab
        prompts.append(p)
    return prompts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF repo dir (config.json + tokenizer.json)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt-json", help='JSON file with {"prompt": text}')
    src.add_argument("--synthetic", action="store_true", help="random token ids, fixed seed")
    p.add_argument("--lens", default=DEFAULT_LENS, help="comma-separated prompt lengths")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--tf-prefill-chunk", type=int, default=None,
                   help="transformer chunked-prefill size (ignored by lastkv)")
    p.add_argument("--new", type=int, default=1, help="decode tokens after prefill")
    p.add_argument("--out", help="also write the JSON summary to this path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("bench_prefill.py needs a CUDA GPU")
    lens = parse_int_list(args.lens)
    if not lens:
        raise SystemExit("--lens is empty")
    vocab = read_vocab_size(args.model)
    if args.batch > vocab:
        raise SystemExit("--batch exceeds vocab; unique first tokens are required")
    ids = base_token_ids(args, max(lens), vocab)
    max_seq_len = max(lens) + args.new + 8

    from lastkv.flashinfer.engine import Engine

    t0 = time.perf_counter()
    engine = Engine(
        args.model,
        max_batch_size=args.batch,
        max_seq_len=max_seq_len,
        page_size=args.page_size,
        tf_prefill_chunk=args.tf_prefill_chunk,
    )
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    pool_gib = engine.pool_bytes() / GIB

    print(
        f"model={args.model} family={engine.family} batch={args.batch} "
        f"new={args.new} page={args.page_size} trials={args.trials} "
        f"load={load_s:.1f}s pool={pool_gib:.2f} GiB"
    )

    # Warmup on the shortest length (excluded from timing).
    warm = build_prompts(ids, lens[0], args.batch, vocab)
    engine.generate(warm, max_new_tokens=args.new, temperature=0.0)
    torch.cuda.synchronize()

    results = []
    header = (
        f"{'len':>8} {'trial':>5} {'prefill_s':>10} {'prefill_tok/s':>14} "
        f"{'peak_GiB':>9} {'kv_pre_GiB':>10} {'pool_GiB':>9}"
    )
    print(header)
    for length in lens:
        prompts = build_prompts(ids, length, args.batch, vocab)
        trials = []
        for _ in range(args.trials):
            engine.generate(prompts, max_new_tokens=args.new, temperature=0.0)
            s = engine.stats
            prefill_s = float(s["prefill_seconds"])
            trials.append({
                "prefill_s": prefill_s,
                "prefill_tok_s": args.batch * length / prefill_s,
                "decode_s": float(s["decode_seconds"]),
                "peak_gib": float(s["peak_memory_bytes"]) / GIB,
                "kv_after_prefill_gib": float(s["kv_bytes_after_prefill"]) / GIB,
                "pool_gib": float(s["pool_bytes"]) / GIB,
            })
        med_s = statistics.median(t["prefill_s"] for t in trials)
        med_tok_s = args.batch * length / med_s
        peak_gib = max(t["peak_gib"] for t in trials)
        for i, t in enumerate(trials, start=1):
            print(
                f"{length:>8} {i:>5} {t['prefill_s']:>10.3f} {t['prefill_tok_s']:>14.1f} "
                f"{t['peak_gib']:>9.2f} {t['kv_after_prefill_gib']:>10.2f} "
                f"{t['pool_gib']:>9.2f}"
            )
        print(
            f"{length:>8} {'med':>5} {med_s:>10.3f} {med_tok_s:>14.1f} "
            f"{peak_gib:>9.2f} {trials[0]['kv_after_prefill_gib']:>10.2f} "
            f"{trials[0]['pool_gib']:>9.2f}"
        )
        results.append({
            "len": length,
            "prefill_s_median": med_s,
            "prefill_tok_s": med_tok_s,
            "peak_gib": peak_gib,
            "kv_after_prefill_gib": trials[0]["kv_after_prefill_gib"],
            "pool_gib": trials[0]["pool_gib"],
            "per_trial": trials,
        })

    summary = {
        "script": "bench_prefill",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0),
        "model": args.model,
        "family": engine.family,
        "source": "prompt_json" if args.prompt_json else "synthetic",
        "batch": args.batch,
        "new_tokens": args.new,
        "page_size": args.page_size,
        "tf_prefill_chunk": args.tf_prefill_chunk,
        "max_seq_len": max_seq_len,
        "trials": args.trials,
        "load_s": round(load_s, 3),
        "pool_gib": round(pool_gib, 3),
        "results": results,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

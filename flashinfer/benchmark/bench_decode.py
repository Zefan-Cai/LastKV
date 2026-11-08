"""Steady-state decode cost vs number of resident sequences N.

For each N a fresh engine is built with ``max_batch_size=N`` (the pool is sized
by N, so the old one is freed first). Each trial prefills N prompts once and
decodes ``--gen`` tokens with ``record_step_times=True``; the first 8 decode
steps are dropped as warmup. Reports median step ms, aggregate / per-sequence
tok/s, memory, and a least-squares fit t_step = a + b*N.

Prints a per-trial table and exactly one JSON object as the last stdout line.
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
WARMUP_STEPS = 8


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def read_vocab_size(model_path: str) -> int:
    with open(os.path.join(model_path, "config.json")) as f:
        return int(json.load(f)["vocab_size"])


def base_token_ids(args: argparse.Namespace, length: int, vocab: int) -> list[int]:
    if args.prompt_json:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
        with open(args.prompt_json) as f:
            text = json.load(f)["prompt"]
        ids = tok.encode(text, add_special_tokens=False).ids
        if len(ids) < length:
            raise SystemExit(
                f"--prompt-json tokenises to {len(ids)} tokens, need {length}"
            )
        return ids[:length]
    rng = random.Random(SEED)
    return [rng.randrange(vocab) for _ in range(length)]


def build_prompts(ids: list[int], batch: int, vocab: int) -> list[list[int]]:
    prompts = []
    for i in range(batch):
        p = list(ids)
        p[0] = (p[0] + i) % vocab  # unique first token per request
        prompts.append(p)
    return prompts


def fit_linear(xs: list[float], ys: list[float]) -> dict | None:
    n = len(xs)
    if n == 0:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return {"a": my, "b": 0.0, "r2": 1.0 if len(set(ys)) == 1 else 0.0, "n": n}
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {"a": a, "b": b, "r2": r2, "n": n}


def run_trial(engine, prompts: list[list[int]], gen: int, batch: int) -> dict:
    engine.generate(prompts, max_new_tokens=gen, temperature=0.0, record_step_times=True)
    s = engine.stats
    step_ms = [float(ms) for ms, _rows in s.get("step_ms", [])]
    kept = step_ms[WARMUP_STEPS:]
    if not kept:
        raise SystemExit(
            f"--gen {gen} leaves no decode steps after dropping {WARMUP_STEPS}"
        )
    med = statistics.median(kept)
    return {
        "step_ms_median": med,
        "step_ms_mean": statistics.fmean(kept),
        "decode_tok_s": batch * 1000.0 / med,
        "decode_tok_s_per_seq": 1000.0 / med,
        "steps": len(kept),
        "prefill_s": float(s["prefill_seconds"]),
        "decode_s": float(s["decode_seconds"]),
        "peak_gib": float(s["peak_memory_bytes"]) / GIB,
        "pool_gib": float(s["pool_bytes"]) / GIB,
        "kv_gib": float(s["kv_bytes_end"]) / GIB,
        "kv_after_prefill_gib": float(s["kv_bytes_after_prefill"]) / GIB,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt-json", help='JSON file with {"prompt": text}')
    src.add_argument("--synthetic", action="store_true", help="random token ids, fixed seed")
    p.add_argument("--len", type=int, default=126976, help="context length per sequence")
    p.add_argument("--batches", default="1,4,8,16,32", help="comma-separated N values")
    p.add_argument("--gen", type=int, default=64)
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--tf-prefill-chunk", type=int, default=None)
    p.add_argument("--out", help="also write the JSON summary to this path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("bench_decode.py needs a CUDA GPU")
    batches = parse_int_list(args.batches)
    if not batches:
        raise SystemExit("--batches is empty")
    if args.gen <= WARMUP_STEPS:
        raise SystemExit(f"--gen must exceed the {WARMUP_STEPS}-step warmup")
    vocab = read_vocab_size(args.model)
    if max(batches) > vocab:
        raise SystemExit("max batch exceeds vocab; unique first tokens are required")
    ids = base_token_ids(args, args.len, vocab)
    max_seq_len = args.len + args.gen + 8

    from lastkv.flashinfer.engine import Engine

    print(
        f"model={args.model} len={args.len} gen={args.gen} trials={args.trials} "
        f"page={args.page_size} batches={batches} warmup_steps={WARMUP_STEPS}"
    )
    print(
        f"{'N':>5} {'trial':>5} {'step_ms':>9} {'tok/s':>10} {'tok/s/seq':>10} "
        f"{'peak_GiB':>9} {'pool_GiB':>9} {'kv_GiB':>8} {'prefill_s':>10}"
    )

    results = []
    for batch in batches:
        prompts = build_prompts(ids, batch, vocab)
        engine = None
        try:
            t0 = time.perf_counter()
            engine = Engine(
                args.model,
                max_batch_size=batch,
                max_seq_len=max_seq_len,
                page_size=args.page_size,
                tf_prefill_chunk=args.tf_prefill_chunk,
            )
            torch.cuda.synchronize()
            load_s = time.perf_counter() - t0
            trials = [run_trial(engine, prompts, args.gen, batch) for _ in range(args.trials)]
            med = statistics.median(t["step_ms_median"] for t in trials)
            peak_gib = max(t["peak_gib"] for t in trials)
            entry = {
                "batch": batch,
                "status": "ok",
                "family": engine.family,
                "load_s": round(load_s, 3),
                "step_ms_median": med,
                "decode_tok_s": batch * 1000.0 / med,
                "decode_tok_s_per_seq": 1000.0 / med,
                "peak_gib": peak_gib,
                "pool_gib": trials[0]["pool_gib"],
                "kv_gib": trials[0]["kv_gib"],
                "kv_after_prefill_gib": trials[0]["kv_after_prefill_gib"],
                "per_trial": trials,
            }
            for i, t in enumerate(trials, start=1):
                print(
                    f"{batch:>5} {i:>5} {t['step_ms_median']:>9.3f} "
                    f"{t['decode_tok_s']:>10.1f} {t['decode_tok_s_per_seq']:>10.2f} "
                    f"{t['peak_gib']:>9.2f} {t['pool_gib']:>9.2f} "
                    f"{t['kv_gib']:>8.3f} {t['prefill_s']:>10.2f}"
                )
            print(
                f"{batch:>5} {'med':>5} {med:>9.3f} {entry['decode_tok_s']:>10.1f} "
                f"{entry['decode_tok_s_per_seq']:>10.2f} {peak_gib:>9.2f} "
                f"{entry['pool_gib']:>9.2f} {entry['kv_gib']:>8.3f} "
                f"{trials[0]['prefill_s']:>10.2f}"
            )
        except torch.cuda.OutOfMemoryError as exc:
            entry = {"batch": batch, "status": "oom", "error": str(exc)[:200]}
            print(f"{batch:>5} {'-':>5} {'oom':>9}")
        finally:
            del engine
            torch.cuda.empty_cache()
        results.append(entry)

    ok = [r for r in results if r["status"] == "ok"]
    fit = fit_linear(
        [float(r["batch"]) for r in ok],
        [float(r["step_ms_median"]) for r in ok],
    )
    if fit is not None:
        print(
            f"fit: step_ms = {fit['a']:.4f} + {fit['b']:.6f} * N "
            f"(r2={fit['r2']:.4f}, n={fit['n']})"
        )

    summary = {
        "script": "bench_decode",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0),
        "model": args.model,
        "source": "prompt_json" if args.prompt_json else "synthetic",
        "len": args.len,
        "gen": args.gen,
        "warmup_steps": WARMUP_STEPS,
        "trials": args.trials,
        "page_size": args.page_size,
        "tf_prefill_chunk": args.tf_prefill_chunk,
        "max_seq_len": max_seq_len,
        "results": results,
        "fit": fit,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

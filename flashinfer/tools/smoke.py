"""Smoke test: load a checkpoint, greedy-generate a few tokens, print timings.

    python tools/smoke.py --model /path/to/ckpt --prompt-json p1k.json --tokens 1500 --new 48
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-json")
    p.add_argument("--tokens", type=int, default=1500)
    p.add_argument("--new", type=int, default=48)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--tf-prefill-chunk", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    args = p.parse_args()

    from lastkv.flashinfer.engine import Engine
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    if args.prompt_json:
        text = json.load(open(args.prompt_json))["prompt"]
        ids = tok.encode(text, add_special_tokens=False).ids[: args.tokens]
    else:
        import random

        rng = random.Random(1234)
        ids = [rng.randrange(120000) for _ in range(args.tokens)]  # synthetic, seeded
    max_seq_len = args.max_seq_len or (len(ids) + args.new + 8)
    t0 = time.perf_counter()
    eng = Engine(args.model, max_batch_size=args.batch, max_seq_len=max_seq_len, page_size=args.page_size, tf_prefill_chunk=args.tf_prefill_chunk)
    torch.cuda.synchronize()
    print(f"family={eng.family} load {time.perf_counter() - t0:.1f}s prompt_tokens={len(ids)} pool={eng.pool_bytes() / 2**30:.2f} GiB")
    prompts = [list(ids) for _ in range(args.batch)]
    for i in range(1, args.batch):
        prompts[i][0] = (prompts[i][0] + i) % eng.config.vocab_size  # unique first token
    # Discarded warmup (JIT module load, first plans, chunk commits) so the printed timings are steady-state.
    eng.generate([p[: min(len(p), 2 * getattr(eng, "chunk", 1024) + 64)] for p in prompts], max_new_tokens=4, temperature=0.0)
    outs = eng.generate(prompts, max_new_tokens=args.new, temperature=0.0, record_step_times=True)
    s = eng.stats
    steps = max(1, int(s["decode_steps"]))
    print(f"prefill {s['prefill_seconds']:.3f}s ({len(ids) * args.batch / s['prefill_seconds']:.0f} tok/s) | "
          f"decode {s['decode_seconds']:.3f}s = {steps} steps, {1000 * s['decode_seconds'] / steps:.2f} ms/step, "
          f"{args.batch * steps / s['decode_seconds']:.1f} tok/s | peak {s['peak_memory_bytes'] / 2**30:.2f} GiB | "
          f"kv {s['kv_bytes_end'] / 2**30:.3f} GiB")
    if "step_ms" in s and s["step_ms"]:
        ms = sorted(t for t, _ in s["step_ms"])
        print(f"step ms: min {ms[0]:.2f} med {ms[len(ms) // 2]:.2f} max {ms[-1]:.2f}")
    print("output:", repr(tok.decode(outs[0].token_ids)))


if __name__ == "__main__":
    main()

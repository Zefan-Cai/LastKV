"""Mixed-batch consistency check.

Prompts of different lengths are greedy-decoded separately (batch 1) and
together (one batch). With the default lengths 1000 / 2024 / 3048 / 2500 and
48 new tokens (47 decode steps), the joint run exercises ragged chunk steps,
rows with and without older history in the same decode step (the -inf LSE
mask), a row crossing n_done 0 -> 1 during decode, two rows committing in the
same decode step with different chunk indices, and a row that never commits.
GEMM shapes differ between the runs, so bf16 rounding may eventually flip a
near-tie; we report the first divergence index per row and require a common
prefix of at least --min-match tokens. On H100 with the 8B checkpoint, batches
of four identical 2,024-token rows (distinct first token) diverge at token 18
and 36 in two of the four rows while two-row batches match 48/48, which is the
signature of batch-shape rounding rather than bookkeeping; a bookkeeping bug
diverges within the first few tokens. Numerics are gated by tools/parity_hf.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def first_divergence(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-json", required=True)
    p.add_argument("--lens", default="1000,2024,3048,2500",
                   help="prompt lengths; the default mixes rows with/without older history and decode-time chunk commits")
    p.add_argument("--new", type=int, default=48)
    p.add_argument("--min-match", type=int, default=16)
    p.add_argument("--page-size", type=int, default=16)
    args = p.parse_args()

    from tokenizers import Tokenizer
    from lastkv.flashinfer.engine import Engine

    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    ids = tok.encode(json.load(open(args.prompt_json))["prompt"], add_special_tokens=False).ids
    lens = [int(x) for x in args.lens.split(",")]
    if min(lens) < 1 or len(ids) < max(lens):
        raise SystemExit(f"prompt has {len(ids)} tokens; need every --lens in [1, {len(ids)}]")
    eng = Engine(args.model, max_batch_size=len(lens), max_seq_len=max(lens) + args.new + 8, page_size=args.page_size)
    prompts = []
    for i, n in enumerate(lens):
        pr = ids[:n]
        pr[0] = (pr[0] + i) % eng.config.vocab_size  # unique first token
        prompts.append(pr)
    single = [eng.generate([pr], max_new_tokens=args.new, temperature=0.0)[0].token_ids for pr in prompts]
    joint = [o.token_ids for o in eng.generate(prompts, max_new_tokens=args.new, temperature=0.0)]
    ok = True
    report = []
    for i, (s, j) in enumerate(zip(single, joint)):
        d = first_divergence(s, j)
        report.append({"row": i, "prompt_len": lens[i], "first_divergence": d, "new": args.new})
        print(f"row {i} len {lens[i]}: single vs joint first divergence at {d}/{args.new}")
        print("  single:", repr(tok.decode(s))[:160])
        print("  joint :", repr(tok.decode(j))[:160])
        ok &= d >= args.min_match
    print(json.dumps({"script": "batch_check", "family": eng.family, "ok": ok, "rows": report}))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

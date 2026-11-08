"""Mixed-batch consistency check.

Two prompts of different lengths are greedy-decoded separately (batch 1) and
together (batch 2); the joint run exercises ragged chunk steps, rows with and
without older history in the same decode step (the -inf LSE mask), and chunk
boundaries reached at different steps. GEMM shapes differ between the runs, so
bf16 rounding may eventually flip a near-tie; we report the first divergence
index per row and require a long common prefix.
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
    p.add_argument("--lens", default="900,2500,3100",
                   help="prompt lengths; the default mixes a row without older history (no source pass) with rows that have it")
    p.add_argument("--new", type=int, default=48)
    p.add_argument("--min-match", type=int, default=24)
    p.add_argument("--page-size", type=int, default=16)
    args = p.parse_args()

    from tokenizers import Tokenizer
    from lastkv.flashinfer.engine import Engine

    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    ids = tok.encode(json.load(open(args.prompt_json))["prompt"], add_special_tokens=False).ids
    lens = [int(x) for x in args.lens.split(",")]
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

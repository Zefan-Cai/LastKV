"""Teacher-forced parity gate: FlashInfer engine vs the HF reference implementation.

For a LastKV checkpoint the reference is ``lastkv.llm`` (needs ``PYTHONPATH`` to
include ``LastKV/llm/src`` and transformers 4.57.6); for a Llama checkpoint it
is ``transformers.LlamaForCausalLM``. Both run bf16 / SDPA on the same GPU.

The prompt is tokenised from a bench prompt JSON (``{"prompt": text}``) or
synthesised from a seed. ``--prefill-tokens`` splits the sequence: the prefix
goes through the engine's prefill path, the remainder is teacher-forced one
token per decode step, so both paths are gated. Metrics: mean / p99 |dlogprob|
of the next-token log-probabilities, top-1 agreement, per-path and per-chunk
breakdown, optional greedy continuation match. Prints one JSON line last.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-json", help='bench prompt file {"prompt": text}; tokenised with tokenizer.json')
    p.add_argument("--tokens", type=int, default=5120, help="sequence length (truncate / synthesize)")
    p.add_argument("--prefill-tokens", type=int, default=None, help="prefix length through prefill; rest teacher-forced decode")
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--greedy", type=int, default=0, help="also compare N greedy continuation tokens")
    p.add_argument("--ref-cache", help="path to save/load the HF reference logprobs (.pt)")
    p.add_argument("--tf-prefill-chunk", type=int, default=None, help="transformer: chunked prefill of C tokens")
    p.add_argument("--out", help="write the JSON summary here too")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def load_tokens(args, vocab_size: int) -> list[int]:
    if args.prompt_json:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
        text = json.load(open(args.prompt_json))["prompt"]
        ids = tok.encode(text, add_special_tokens=False).ids
        if len(ids) < args.tokens:
            raise SystemExit(f"prompt has {len(ids)} tokens < --tokens {args.tokens}")
        return ids[: args.tokens]
    rng = random.Random(args.seed)
    return [rng.randrange(vocab_size) for _ in range(args.tokens)]


@torch.inference_mode()
def hf_reference(model_path: str, family: str, ids: list[int], greedy: int) -> dict:
    device = torch.device("cuda")
    if family == "lastkv":
        from lastkv.llm import CustomRecurrentMarinAKV254ReuseKVForCausalLM as Cls
    else:
        from transformers import LlamaForCausalLM as Cls
    t0 = time.perf_counter()
    model = Cls.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    load_s = time.perf_counter() - t0
    x = torch.tensor([ids], dtype=torch.long, device=device)
    target = x[0, 1:]
    out = model(input_ids=x, use_cache=False)
    logits = out.logits[0]  # [n, V] bf16
    n = len(ids)
    logprob = torch.empty(n - 1, dtype=torch.float32, device=device)
    argmax = torch.empty(n - 1, dtype=torch.long, device=device)
    for s in range(0, n - 1, 2048):
        e = min(n - 1, s + 2048)
        lp = torch.log_softmax(logits[s:e].float(), dim=-1)
        logprob[s:e] = lp.gather(1, target[s:e, None]).squeeze(1)
        argmax[s:e] = lp.argmax(dim=-1)
    del logits, out
    greedy_ids: list[int] = []
    if greedy > 0:
        gen = model.generate(x, max_new_tokens=greedy, do_sample=False, min_new_tokens=greedy, pad_token_id=0)
        greedy_ids = gen[0, n:].tolist()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"logprob": logprob.cpu(), "argmax": argmax.cpu(), "greedy": greedy_ids, "load_s": load_s}


def main() -> None:
    args = parse_args()
    from lastkv.flashinfer.config import ModelConfig

    cfg = ModelConfig.from_model_dir(args.model)
    ids = load_tokens(args, cfg.vocab_size)
    n = len(ids)

    ref = None
    if args.ref_cache and os.path.exists(args.ref_cache):
        ref = torch.load(args.ref_cache)
        if ref.get("ids") != ids or ref.get("model") != os.path.abspath(args.model):
            print("ref cache ids/model differ; recomputing", file=sys.stderr)
            ref = None
    if ref is None:
        ref = hf_reference(args.model, cfg.family, ids, args.greedy)
        ref["ids"] = ids
        ref["model"] = os.path.abspath(args.model)
        if args.ref_cache:
            torch.save(ref, args.ref_cache)

    from lastkv.flashinfer.engine import Engine

    t0 = time.perf_counter()
    engine = Engine(
        args.model,
        max_batch_size=1,
        max_seq_len=n + max(args.greedy, 0) + 8,
        page_size=args.page_size,
        tf_prefill_chunk=args.tf_prefill_chunk,
    )
    engine_load_s = time.perf_counter() - t0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ours = engine.score(ids, prefill_tokens=args.prefill_tokens)
    score_s = time.perf_counter() - t0

    d = (ours["logprob"] - ref["logprob"]).abs()
    agree = (ours["argmax"] == ref["argmax"]).float()
    path = ours["path"]
    pre = torch.tensor([p == "prefill" for p in path])
    chunk = cfg.chunk_size if cfg.family == "lastkv" else 1024
    per_chunk = []
    for c in range(0, n - 1, chunk):
        e = min(n - 1, c + chunk)
        per_chunk.append({
            "pos": [c, e],
            "mean_abs_dlogprob": round(d[c:e].mean().item(), 5),
            "top1_agree": round(agree[c:e].mean().item(), 5),
        })
    summary = {
        "script": "parity_hf",
        "model": args.model,
        "family": cfg.family,
        "tokens": n,
        "prefill_tokens": ours["prefill_tokens"],
        "page_size": args.page_size,
        "tf_prefill_chunk": args.tf_prefill_chunk,
        "mean_abs_dlogprob": round(d.mean().item(), 6),
        "p99_abs_dlogprob": round(d.quantile(0.99).item(), 5),
        "max_abs_dlogprob": round(d.max().item(), 4),
        "top1_agree": round(agree.mean().item(), 5),
        "prefill_path": {
            "n": int(pre.sum()),
            "mean_abs_dlogprob": round(d[pre].mean().item(), 6) if pre.any() else None,
            "top1_agree": round(agree[pre].mean().item(), 5) if pre.any() else None,
        },
        "decode_path": {
            "n": int((~pre).sum()),
            "mean_abs_dlogprob": round(d[~pre].mean().item(), 6) if (~pre).any() else None,
            "top1_agree": round(agree[~pre].mean().item(), 5) if (~pre).any() else None,
        },
        "per_chunk": per_chunk,
        "hf_load_s": round(ref.get("load_s", 0.0), 2),
        "engine_load_s": round(engine_load_s, 2),
        "score_s": round(score_s, 3),
    }
    if args.greedy > 0:
        outs = engine.generate([ids], max_new_tokens=args.greedy, temperature=0.0)
        g = outs[0].token_ids
        match = 0
        for a, b in zip(g, ref["greedy"]):
            if a != b:
                break
            match += 1
        summary["greedy"] = {"n": args.greedy, "prefix_match": match, "ours": g[:16], "hf": ref["greedy"][:16]}
    print(json.dumps({k: v for k, v in summary.items() if k != "per_chunk"}, indent=1))
    print("per_chunk:", json.dumps(per_chunk))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

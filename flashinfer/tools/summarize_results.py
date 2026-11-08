"""Render the benchmark / parity JSON files of a results directory as Markdown.

    python tools/summarize_results.py /path/to/results > docs/RESULTS_H100.md
"""

from __future__ import annotations

import glob
import json
import os
import sys


def load(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def fmt(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    return f"{x:,.{nd}f}"


def prefill_section(d: str) -> list[str]:
    out = ["## Single-stream prefill (batch 1, same engine, H100)", ""]
    tf = load(os.path.join(d, "prefill-tf-b1.json"))
    lk = load(os.path.join(d, "prefill-lastkv-b1.json"))
    tfc = load(os.path.join(d, "prefill-tf-chunked1024-b1.json"))
    if not (tf and lk):
        return out + ["(missing prefill-tf-b1.json / prefill-lastkv-b1.json)", ""]
    by = lambda r: {x["len"]: x for x in r["results"]}
    T, L = by(tf), by(lk)
    C = by(tfc) if tfc else {}
    out += ["| prompt tokens | Transformer s | LastKV s | Transformer tok/s | LastKV tok/s | LastKV/Transformer | Transformer chunked-1024 tok/s | LastKV KV GiB | Transformer KV GiB |",
            "|---|---|---|---|---|---|---|---|---|"]
    for n in sorted(set(T) & set(L)):
        t, l = T[n], L[n]
        c = C.get(n)
        out.append(f"| {n:,} | {t['prefill_s_median']:.3f} | {l['prefill_s_median']:.3f} | {fmt(t['prefill_tok_s'], 0)} | {fmt(l['prefill_tok_s'], 0)} | "
                   f"{l['prefill_tok_s'] / t['prefill_tok_s']:.3f} | {fmt(c['prefill_tok_s'], 0) if c else '-'} | {l['kv_after_prefill_gib']:.2f} | {t['kv_after_prefill_gib']:.2f} |")
    out.append("")
    # batched prefill at one length
    rows = []
    for path in sorted(glob.glob(os.path.join(d, "prefill-*-b[0-9]*.json"))):
        r = load(path)
        if not r or r.get("batch", 1) == 1:
            continue
        for x in r["results"]:
            fam = "lastkv" if "lastkv" in os.path.basename(path) else "transformer"
            if fam == "transformer" and not r.get("tf_prefill_chunk"):
                fam = "transformer (serial full-prompt)"
            rows.append((fam, r["batch"], x))
    if rows:
        out += ["### Simultaneous prefill of N prompts (static batch, chunk-synchronous for LastKV)", "",
                "| family | N | tokens each | wall s (median) | aggregate tok/s | per-prompt tok/s | peak GiB | pool GiB |", "|---|---|---|---|---|---|---|---|"]
        for fam, b, x in sorted(rows, key=lambda r: (r[0], r[2]["len"], r[1])):
            out.append(f"| {fam} | {b} | {x['len']:,} | {x['prefill_s_median']:.2f} | {fmt(x['prefill_tok_s'], 0)} | {fmt(x['prefill_tok_s'] / b, 0)} | {x['peak_gib']:.1f} | {x['pool_gib']:.1f} |")
        out.append("")
    return out


def decode_section(d: str) -> list[str]:
    out = ["## Steady-state decode versus resident sequences N (fixed context, same engine)", ""]
    files = sorted(glob.glob(os.path.join(d, "decode-*.json")))
    if not files:
        return out + ["(no decode-*.json)", ""]
    groups: dict[int, dict[str, list]] = {}
    fits: dict[tuple, dict] = {}
    for path in files:
        r = load(path)
        if not r:
            continue
        fam = "lastkv" if "lastkv" in os.path.basename(path) else "transformer"
        for x in r["results"]:
            bucket = groups.setdefault(r["len"], {}).setdefault(fam, {})
            prev = bucket.get(x["batch"])
            if prev is None or (prev.get("status") != "ok" and x.get("status") == "ok"):
                bucket[x["batch"]] = x
        if r.get("fit"):
            fits[(r["len"], fam, os.path.basename(path))] = r["fit"]
    for n in sorted(groups):
        out += [f"### context {n:,} tokens", "",
                "| family | N | step ms (median) | aggregate tok/s | per-seq tok/s | KV GiB | pool GiB | peak GiB |", "|---|---|---|---|---|---|---|---|"]
        for fam in ("transformer", "lastkv"):
            for x in sorted(groups[n].get(fam, {}).values(), key=lambda e: e["batch"]):
                if x.get("status") != "ok":
                    out.append(f"| {fam} | {x['batch']} | OOM | | | | | |")
                    continue
                out.append(f"| {fam} | {x['batch']} | {x['step_ms_median']:.2f} | {x['decode_tok_s']:.1f} | {x['decode_tok_s_per_seq']:.2f} | {x['kv_gib']:.2f} | {x['pool_gib']:.1f} | {x['peak_gib']:.1f} |")
        out.append("")
        for (ln, fam, name), f in sorted(fits.items()):
            if ln == n and f and f.get("n", 0) >= 2 and f.get("b", 0) > 0:
                out.append(f"- fit {fam} ({name}): step_ms = {f['a']:.2f} + {f['b']:.3f} * N (r2 {f['r2']:.3f}, n={f['n']}); ceiling 1000/b = {1000 / f['b']:.0f} tok/s")
        out.append("")
    return out


def plan_section(d: str) -> list[str]:
    r = load(os.path.join(d, "plan-bench.json"))
    out = ["## FlashInfer decode plan() host cost (no model; 32 dummy layers)", ""]
    if not r:
        return out + ["(no plan-bench.json)", ""]
    out += [f"context {r.get('len'):,} tokens, page {r.get('page_size')}, {r.get('q_heads')}/{r.get('kv_heads')} heads x {r.get('head_dim')}", "",
            "| N | plan ms, CPU indptr + device indices (idle / busy) | plan ms, all inputs on device (idle / busy) | 32-layer step wall ms | step GPU ms |", "|---|---|---|---|---|"]
    for x in r.get("results", []):
        if x.get("status", "ok") != "ok":
            out.append(f"| {x.get('batch')} | {x.get('status')} | | | |")
            continue
        def pm(key):
            v = x.get(key) or {}
            return f"{v.get('median_ms', 0):.3f}"
        out.append(f"| {x['batch']} | {pm('plan_cpu_inputs_idle')} / {pm('plan_cpu_inputs_busy')} | {pm('plan_all_device_idle')} / {pm('plan_all_device_busy')} | "
                   f"{(x.get('step') or {}).get('wall_ms', 0):.2f} | {(x.get('step') or {}).get('gpu_ms', 0):.2f} |")
    return out + [""]


def parity_section(d: str) -> list[str]:
    out = ["## Teacher-forced parity versus the HF reference", "",
           "| file | family | tokens | prefill tokens | mean abs dlogprob | p99 | top-1 | prefill path | decode path | greedy |", "|---|---|---|---|---|---|---|---|---|---|"]
    for path in sorted(glob.glob(os.path.join(d, "parity-*.json"))):
        r = load(path)
        if not r:
            continue
        g = r.get("greedy")
        out.append(f"| {os.path.basename(path)} | {r['family']} | {r['tokens']} | {r['prefill_tokens']} | {r['mean_abs_dlogprob']:.4f} | {r['p99_abs_dlogprob']:.3f} | {r['top1_agree']:.4f} | "
                   f"{fmt(r['prefill_path']['mean_abs_dlogprob'], 4)} | {fmt(r['decode_path']['mean_abs_dlogprob'], 4)} | {g['prefix_match'] if g else '-'}/{g['n'] if g else '-'} |")
    return out + [""]


def pd_section(d: str) -> list[str]:
    r = load(os.path.join(d, "pd-transfer.json"))
    out = ["## Prefill/decode hand-off: one request's state, GPU to GPU", ""]
    if not r:
        return out + ["(no pd-transfer.json)", ""]
    out += [f"peer access: {r.get('peer_access')}", "", "| tokens | LastKV GB | Transformer GB | ratio | LastKV ms | Transformer ms | GB/s |", "|---|---|---|---|---|---|---|"]
    for x in r["rows"]:
        out.append(f"| {x['tokens']:,} | {x['lastkv_gb']:.3f} | {x['transformer_gb']:.3f} | {x['bytes_ratio']} | {x['lastkv_ms']} | {x['transformer_ms']} | {x['lastkv_gbps']} |")
    return out + [""]


def main() -> None:
    d = sys.argv[1] if len(sys.argv) > 1 else "results"
    lines = ["# LastKV on FlashInfer — H100 results", "",
             "Standalone eager engine, one H100 80GB per run, bf16, page 16; LastKV = marin SFT 8B checkpoint (C=1024), Transformer = Llama-3.1-8B; token-id prompts from the same 128K bench prompt, unique first token per request.", ""]
    lines += prefill_section(d) + decode_section(d) + plan_section(d) + parity_section(d) + pd_section(d)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

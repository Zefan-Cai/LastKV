"""Isolate the FlashInfer paged-decode kernel: does it respect last_page_len?

Fills the whole paged pool with NaN, writes real K/V only into the first
``kv_len`` slots, runs the decode wrapper (tensor-core and CUDA-core paths) and
the paged prefill wrapper with qo_len=1, and compares with a torch reference
over exactly ``kv_len`` tokens. A NaN or large error means the kernel read slots
beyond ``last_page_len`` (or our page bookkeeping is wrong).
"""

from __future__ import annotations

import json
import sys

import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper

PAGE, HKV, HQ, D = 16, 8, 32, 128


def reference(q, k, v):
    # q [HQ, D]; k, v [T, HKV, D]
    g = HQ // HKV
    kq = k.repeat_interleave(g, dim=1).float()  # [T, HQ, D]
    vq = v.repeat_interleave(g, dim=1).float()
    s = torch.einsum("hd,thd->ht", q.float(), kq) / D**0.5
    p = torch.softmax(s, dim=-1)
    return torch.einsum("ht,thd->hd", p, vq)


def main() -> None:
    dev = torch.device("cuda")
    torch.manual_seed(0)
    lens = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else
            "16,17,1477,1480,2497,2501,2505,2512,4095,4096,4097,4100,4104,4109,4112,5119".split(","))]
    ws = [torch.zeros(256 << 20, dtype=torch.uint8, device=dev) for _ in range(3)]
    dec_tc = BatchDecodeWithPagedKVCacheWrapper(ws[0], "NHD", use_tensor_cores=True)
    dec_cc = BatchDecodeWithPagedKVCacheWrapper(ws[1], "NHD", use_tensor_cores=False)
    pre = BatchPrefillWithPagedKVCacheWrapper(ws[2], "NHD")
    max_pages = (max(lens) + PAGE - 1) // PAGE + 2
    results = []
    for kv_len in lens:
        k_pool = torch.full((max_pages, PAGE, HKV, D), float("nan"), dtype=torch.bfloat16, device=dev)
        v_pool = torch.full_like(k_pool, float("nan"))
        k = torch.randn(kv_len, HKV, D, dtype=torch.bfloat16, device=dev)
        v = torch.randn(kv_len, HKV, D, dtype=torch.bfloat16, device=dev)
        k_pool.view(-1, HKV, D)[:kv_len] = k
        v_pool.view(-1, HKV, D)[:kv_len] = v
        q = torch.randn(1, HQ, D, dtype=torch.bfloat16, device=dev)
        pages = (kv_len + PAGE - 1) // PAGE
        last = (kv_len - 1) % PAGE + 1
        indptr = torch.tensor([0, pages], dtype=torch.int32)
        indices = torch.arange(pages, dtype=torch.int32, device=dev)
        last_t = torch.tensor([last], dtype=torch.int32)
        ref = reference(q[0], k, v)
        row = {"kv_len": kv_len, "pages": pages, "last_page_len": last}
        for name, w in (("decode_tc", dec_tc), ("decode_cc", dec_cc)):
            w.plan(indptr, indices, last_t, HQ, HKV, D, PAGE, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
            out, lse = w.run(q, (k_pool, v_pool), return_lse=True)
            err = (out[0].float() - ref).abs().max().item()
            row[name] = {"max_abs_err": round(err, 4), "nan": bool(torch.isnan(out).any().item()), "lse_nan": bool(torch.isnan(lse).any().item())}
        qo = torch.tensor([0, 1], dtype=torch.int32)
        pre.plan(qo, indptr, indices, last_t, HQ, HKV, D, PAGE, causal=True, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
        out, lse = pre.run(q, (k_pool, v_pool), return_lse=True)
        err = (out[0].float() - ref).abs().max().item()
        row["prefill_q1"] = {"max_abs_err": round(err, 4), "nan": bool(torch.isnan(out).any().item())}
        print(row)
        results.append(row)
    torch.cuda.synchronize()
    print(json.dumps({"script": "kernel_check", "rows": results}))


if __name__ == "__main__":
    main()

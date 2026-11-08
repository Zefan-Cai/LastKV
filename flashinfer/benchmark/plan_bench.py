"""Host cost of FlashInfer ``plan()`` in this engine's decode configuration.

Times BatchDecodeWithPagedKVCacheWrapper (NHD, tensor cores, 256 MB workspace)
per batch N at one context length: (a) CPU indptr/last_page_len + device
indices (the engine's convention), (b) all inputs on device (the vLLM-style
call that can force a D2H sync), (c) both again with a background matmul loop
keeping the GPU busy, and (d) one plan + 32-layer run() over a dummy paged KV
cache. No weights are loaded. Table then one JSON object as the last line.

    python benchmark/plan_bench.py --batches 1,4,8,16,32 --len 126976
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import threading
import time

import torch


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))]


def time_calls(fn, reps: int, warmup: int) -> dict:
    """Median/p90 host wall time (ms) around ``fn``; sync only at the edges."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(1000.0 * (time.perf_counter() - t0))
    torch.cuda.synchronize()
    return {"median_ms": statistics.median(samples), "p90_ms": percentile(samples, 0.90)}


class BusyGPU:
    """Bounded-queue matmul load on a side stream.

    Depth is capped so a default-stream sync waits for a few ms, like a server
    whose GPU is busy with other requests, not for an unbounded backlog.
    """

    def __init__(self, size: int = 8192, depth: int = 4) -> None:
        self._stop = threading.Event()
        self._started = threading.Event()
        self.stream = torch.cuda.Stream()
        dev = torch.device("cuda")
        self.a = torch.randn(size, size, device=dev, dtype=torch.bfloat16) * 0.01
        self.b = torch.randn(size, size, device=dev, dtype=torch.bfloat16) * 0.01
        self.out = torch.empty(size, size, device=dev, dtype=torch.bfloat16)
        self._depth = depth
        self._events: collections.deque = collections.deque()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        with torch.cuda.stream(self.stream):
            while not self._stop.is_set():
                if len(self._events) >= self._depth:
                    if not self._events[0].query():
                        time.sleep(0.0005)
                        continue
                    self._events.popleft()
                torch.mm(self.a, self.b, out=self.out)
                ev = torch.cuda.Event()
                ev.record()
                self._events.append(ev)
                self._started.set()

    def __enter__(self) -> "BusyGPU":
        self._thread.start()
        self._started.wait()
        time.sleep(0.02)  # let the bounded queue fill
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(timeout=10.0)
        torch.cuda.synchronize()


def make_inputs(batch: int, ctx_len: int, page_size: int, device, total_pages: int):
    pages_per_seq = cdiv(ctx_len, page_size)
    last_len = ctx_len - (pages_per_seq - 1) * page_size
    indptr = torch.tensor([r * pages_per_seq for r in range(batch + 1)], dtype=torch.int32)
    last = torch.full((batch,), last_len, dtype=torch.int32)
    indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    return indptr, indices, last, pages_per_seq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches", default="1,4,8,16,32")
    p.add_argument("--len", type=int, default=126976, help="context length per sequence")
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--q-heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--workspace-mb", type=int, default=256)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--busy-size", type=int, default=8192, help="busy-loop matmul side")
    p.add_argument("--busy-depth", type=int, default=4, help="busy-loop queue depth")
    p.add_argument("--out", help="also write the JSON summary to this path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("plan_bench.py needs a CUDA GPU")

    import flashinfer
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    device = torch.device("cuda")
    dtype = torch.bfloat16
    page = args.page_size
    ws_bytes = args.workspace_mb * 1024 * 1024
    base_args = (args.q_heads, args.kv_heads, args.head_dim, page)
    kw = {"q_data_type": dtype, "kv_data_type": dtype}
    version = getattr(flashinfer, "__version__", "unknown")
    print(
        f"gpu={torch.cuda.get_device_name(0)} flashinfer={version} len={args.len} "
        f"page={page} heads={args.q_heads}/{args.kv_heads}x{args.head_dim} "
        f"layers={args.layers} workspace={args.workspace_mb}MB reps={args.reps}"
    )

    results = []
    for batch in batches:
        pages_per_seq = cdiv(args.len, page)
        total_pages = batch * pages_per_seq
        kv_bytes = 2 * total_pages * page * args.kv_heads * args.head_dim * 2
        k = v = q = None
        engine_w = dev_w = run_w = None
        try:
            shape = (total_pages, page, args.kv_heads, args.head_dim)
            k = torch.zeros(shape, dtype=dtype, device=device)
            v = torch.zeros_like(k)
            q = torch.randn((batch, args.q_heads, args.head_dim), dtype=dtype, device=device) * 0.1
            indptr_cpu, indices_dev, last_cpu, _ = make_inputs(
                batch, args.len, page, device, total_pages
            )
            indptr_dev = indptr_cpu.to(device)
            last_dev = last_cpu.to(device)

            def new_wrapper() -> BatchDecodeWithPagedKVCacheWrapper:
                ws = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)
                return BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)

            engine_w, dev_w, run_w = new_wrapper(), new_wrapper(), new_wrapper()

            # (a) engine convention: CPU indptr/last, device indices.
            plan_engine = lambda: engine_w.plan(indptr_cpu, indices_dev, last_cpu, *base_args, **kw)
            # (b) vLLM style: everything already on device.
            plan_dev = lambda: dev_w.plan(indptr_dev, indices_dev, last_dev, *base_args, **kw)

            idle_engine = time_calls(plan_engine, args.reps, args.warmup)
            idle_dev = time_calls(plan_dev, args.reps, args.warmup)

            # (c) same two variants with the GPU kept busy by a side stream.
            with BusyGPU(args.busy_size, args.busy_depth):
                busy_engine = time_calls(plan_engine, args.reps, args.warmup)
                busy_dev = time_calls(plan_dev, args.reps, args.warmup)

            # (d) one plan + a full decode step over args.layers layers.
            run_w.plan(indptr_cpu, indices_dev, last_cpu, *base_args, **kw)
            for _ in range(args.warmup):
                run_w.run(q, (k, v))
            torch.cuda.synchronize()
            wall, gpu = [], []
            for _ in range(max(1, args.reps // 10)):
                torch.cuda.synchronize()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                t0 = time.perf_counter()
                e0.record()
                for _ in range(args.layers):
                    run_w.run(q, (k, v))
                e1.record()
                torch.cuda.synchronize()
                wall.append(1000.0 * (time.perf_counter() - t0))
                gpu.append(e0.elapsed_time(e1))
            step = {
                "layers": args.layers,
                "wall_ms": statistics.median(wall),
                "gpu_ms": statistics.median(gpu),
            }
            step["gpu_ms_per_layer"] = step["gpu_ms"] / args.layers
            step["gpu_ms_per_layer_vs_plan"] = step["gpu_ms_per_layer"] / idle_engine["median_ms"]

            entry = {
                "batch": batch,
                "status": "ok",
                "pages_per_seq": pages_per_seq,
                "kv_gib": kv_bytes / 2 ** 30,
                "plan_cpu_inputs_idle": idle_engine,
                "plan_all_device_idle": idle_dev,
                "plan_cpu_inputs_busy": busy_engine,
                "plan_all_device_busy": busy_dev,
                "step": step,
            }
            print(
                f"N={batch:>3} pages/seq={pages_per_seq:>5} kv={entry['kv_gib']:>6.2f}GiB | "
                f"idle cpu-in {idle_engine['median_ms']:>7.3f}/{idle_engine['p90_ms']:>7.3f} "
                f"dev-in {idle_dev['median_ms']:>7.3f}/{idle_dev['p90_ms']:>7.3f} | "
                f"busy cpu-in {busy_engine['median_ms']:>8.3f}/{busy_engine['p90_ms']:>8.3f} "
                f"dev-in {busy_dev['median_ms']:>8.3f}/{busy_dev['p90_ms']:>8.3f} | "
                f"step wall {step['wall_ms']:>7.3f} gpu {step['gpu_ms']:>7.3f} ms"
            )
        except torch.cuda.OutOfMemoryError as exc:
            entry = {"batch": batch, "status": "oom", "error": str(exc)[:200]}
            print(f"N={batch:>3} oom")
        finally:
            del k, v, q, engine_w, dev_w, run_w
            torch.cuda.empty_cache()
        results.append(entry)

    summary = {
        "script": "plan_bench",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0), "flashinfer": version,
        "len": args.len, "page_size": page,
        "q_heads": args.q_heads, "kv_heads": args.kv_heads, "head_dim": args.head_dim,
        "layers": args.layers, "workspace_mb": args.workspace_mb,
        "reps": args.reps, "warmup": args.warmup,
        "busy": {"matmul_size": args.busy_size, "queue_depth": args.busy_depth},
        "results": results,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

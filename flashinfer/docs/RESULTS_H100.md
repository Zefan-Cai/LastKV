# LastKV on FlashInfer — H100 results

Standalone eager engine, one H100 80GB per run, bf16, page 16; LastKV = marin SFT 8B checkpoint (C=1024), Transformer = Llama-3.1-8B; token-id prompts from the same 128K bench prompt, unique first token per request.

## Single-stream prefill (batch 1, same engine, H100)

| prompt tokens | Transformer s | LastKV s | Transformer tok/s | LastKV tok/s | LastKV/Transformer | Transformer chunked-1024 tok/s | LastKV KV GiB | Transformer KV GiB |
|---|---|---|---|---|---|---|---|---|
| 8,192 | 0.216 | 0.243 | 38,009 | 33,649 | 0.885 | 35,256 | 0.28 | 1.00 |
| 16,384 | 0.490 | 0.555 | 33,459 | 29,521 | 0.882 | 30,307 | 0.31 | 2.00 |
| 32,768 | 1.185 | 1.367 | 27,663 | 23,966 | 0.866 | 24,269 | 0.38 | 4.00 |
| 65,536 | 3.237 | 3.754 | 20,247 | 17,458 | 0.862 | 17,511 | 0.50 | 8.00 |
| 126,976 | 9.520 | 11.031 | 13,337 | 11,510 | 0.863 | 11,446 | 0.73 | 15.50 |

### Simultaneous prefill of N prompts (static batch, chunk-synchronous for LastKV)

| family | N | tokens each | wall s (median) | aggregate tok/s | per-prompt tok/s | peak GiB | pool GiB |
|---|---|---|---|---|---|---|---|
| lastkv | 4 | 126,976 | 43.54 | 11,666 | 2,916 | 19.7 | 2.9 |
| lastkv | 8 | 126,976 | 87.76 | 11,575 | 1,447 | 23.1 | 5.9 |
| transformer (serial full-prompt) | 4 | 65,536 | 12.94 | 20,254 | 5,063 | 55.8 | 32.0 |
| transformer (serial full-prompt) | 2 | 126,976 | 19.01 | 13,358 | 6,679 | 61.9 | 31.0 |

## Steady-state decode versus resident sequences N (fixed context, same engine)

### context 4,096 tokens

| family | N | step ms (median) | aggregate tok/s | per-seq tok/s | KV GiB | pool GiB | peak GiB |
|---|---|---|---|---|---|---|---|
| transformer | 1 | 7.86 | 127.2 | 127.24 | 0.51 | 0.5 | 17.2 |
| transformer | 4 | 8.90 | 449.6 | 112.41 | 2.03 | 2.0 | 20.0 |
| transformer | 16 | 11.25 | 1422.0 | 88.88 | 8.12 | 8.2 | 31.2 |
| transformer | 32 | 14.05 | 2277.3 | 71.17 | 16.25 | 16.3 | 46.1 |
| transformer | 64 | 20.02 | 3197.6 | 49.96 | 32.49 | 32.6 | 49.4 |
| transformer | 96 | 26.20 | 3664.6 | 38.17 | 48.74 | 48.9 | 65.7 |
| transformer | 112 | 29.31 | 3821.0 | 34.12 | 56.86 | 57.1 | 73.9 |
| lastkv | 1 | 12.28 | 81.4 | 81.44 | 0.27 | 0.3 | 16.7 |
| lastkv | 4 | 13.41 | 298.3 | 74.57 | 1.06 | 1.1 | 17.8 |
| lastkv | 16 | 13.60 | 1176.7 | 73.54 | 4.25 | 4.3 | 22.4 |
| lastkv | 32 | 15.03 | 2129.1 | 66.53 | 8.50 | 8.5 | 28.4 |
| lastkv | 64 | 21.23 | 3014.4 | 47.10 | 17.00 | 17.0 | 40.6 |
| lastkv | 96 | 27.52 | 3488.3 | 36.34 | 25.50 | 25.5 | 52.7 |
| lastkv | 128 | 33.80 | 3786.6 | 29.58 | 34.00 | 34.0 | 64.9 |
| lastkv | 192 | OOM | | | | | |

- fit lastkv (decode-lastkv-4096.json): step_ms = 11.34 + 0.169 * N (r2 0.983, n=7); ceiling 1000/b = 5932 tok/s
- fit transformer (decode-tf-4096-full.json): step_ms = 7.62 + 0.194 * N (r2 1.000, n=3); ceiling 1000/b = 5165 tok/s
- fit transformer (decode-tf-4096.json): step_ms = 7.93 + 0.195 * N (r2 0.993, n=4); ceiling 1000/b = 5134 tok/s

### context 8,192 tokens

| family | N | step ms (median) | aggregate tok/s | per-seq tok/s | KV GiB | pool GiB | peak GiB |
|---|---|---|---|---|---|---|---|
| transformer | 1 | 8.02 | 124.7 | 124.74 | 1.01 | 1.0 | 17.7 |
| transformer | 4 | 9.63 | 415.2 | 103.80 | 4.03 | 4.0 | 22.1 |
| transformer | 16 | 13.99 | 1143.5 | 71.47 | 16.12 | 16.2 | 39.7 |
| transformer | 32 | 19.66 | 1627.9 | 50.87 | 32.25 | 32.3 | 63.1 |
| transformer | 48 | 25.85 | 1856.6 | 38.68 | 48.37 | 48.5 | 65.7 |
| transformer | 56 | 28.67 | 1953.3 | 34.88 | 56.43 | 56.5 | 73.8 |
| lastkv | 1 | 11.79 | 84.8 | 84.84 | 0.28 | 0.3 | 16.7 |
| lastkv | 4 | 12.73 | 314.2 | 78.55 | 1.12 | 1.1 | 17.9 |
| lastkv | 16 | 14.98 | 1067.9 | 66.74 | 4.50 | 4.5 | 22.6 |
| lastkv | 32 | 20.54 | 1557.7 | 48.68 | 9.00 | 9.0 | 28.9 |
| lastkv | 64 | 32.90 | 1945.3 | 30.40 | 18.00 | 18.0 | 41.6 |
| lastkv | 128 | 56.21 | 2277.3 | 17.79 | 36.00 | 36.0 | 66.9 |

- fit lastkv (decode-lastkv-8192.json): step_ms = 10.40 + 0.354 * N (r2 0.997, n=6); ceiling 1000/b = 2824 tok/s
- fit transformer (decode-tf-8192-full.json): step_ms = 8.96 + 0.352 * N (r2 1.000, n=2); ceiling 1000/b = 2842 tok/s
- fit transformer (decode-tf-8192.json): step_ms = 7.93 + 0.369 * N (r2 0.998, n=4); ceiling 1000/b = 2707 tok/s

### context 32,768 tokens

| family | N | step ms (median) | aggregate tok/s | per-seq tok/s | KV GiB | pool GiB | peak GiB |
|---|---|---|---|---|---|---|---|
| transformer | 1 | 9.05 | 110.5 | 110.48 | 4.01 | 4.0 | 23.8 |
| transformer | 4 | 13.86 | 288.7 | 72.17 | 16.03 | 16.0 | 36.1 |
| transformer | 8 | 19.61 | 408.0 | 51.01 | 32.06 | 32.1 | 52.1 |
| transformer | 12 | 25.13 | 477.5 | 39.79 | 48.09 | 48.1 | 68.2 |
| transformer | 14 | OOM | | | | | |
| transformer | 15 | OOM | | | | | |
| transformer | 16 | OOM | | | | | |
| lastkv | 1 | 9.81 | 101.9 | 101.92 | 0.38 | 0.4 | 16.8 |
| lastkv | 4 | 14.50 | 275.9 | 68.97 | 1.50 | 1.5 | 18.2 |
| lastkv | 16 | 31.62 | 506.0 | 31.63 | 6.00 | 6.0 | 24.1 |
| lastkv | 32 | 53.84 | 594.3 | 18.57 | 12.00 | 12.0 | 31.9 |
| lastkv | 64 | 99.46 | 643.5 | 10.05 | 24.00 | 24.0 | 47.6 |
| lastkv | 96 | 144.72 | 663.3 | 6.91 | 36.00 | 36.0 | 63.2 |
| lastkv | 128 | OOM | | | | | |

- fit lastkv (decode-lastkv-32768.json): step_ms = 8.66 + 1.418 * N (r2 1.000, n=5); ceiling 1000/b = 705 tok/s
- fit transformer (decode-tf-32768.json): step_ms = 7.81 + 1.456 * N (r2 0.999, n=4); ceiling 1000/b = 687 tok/s

### context 65,536 tokens

| family | N | step ms (median) | aggregate tok/s | per-seq tok/s | KV GiB | pool GiB | peak GiB |
|---|---|---|---|---|---|---|---|
| transformer | 1 | 10.37 | 96.4 | 96.42 | 8.01 | 8.0 | 31.3 |
| transformer | 2 | 13.66 | 146.4 | 73.19 | 16.02 | 16.0 | 39.8 |
| transformer | 4 | 19.36 | 206.6 | 51.64 | 32.03 | 32.0 | 55.9 |
| transformer | 7 | 27.93 | 250.6 | 35.80 | 56.05 | 56.1 | 75.6 |
| transformer | 8 | OOM | | | | | |
| lastkv | 1 | 11.82 | 84.6 | 84.62 | 0.50 | 0.5 | 16.9 |
| lastkv | 4 | 20.33 | 196.7 | 49.19 | 2.00 | 2.0 | 18.8 |
| lastkv | 16 | 54.12 | 295.6 | 18.48 | 8.00 | 8.0 | 26.1 |
| lastkv | 32 | 98.83 | 323.8 | 10.12 | 16.00 | 16.0 | 36.0 |
| lastkv | 64 | 188.42 | 339.7 | 5.31 | 32.00 | 32.0 | 55.6 |

- fit lastkv (decode-lastkv-65536.json): step_ms = 9.13 + 2.802 * N (r2 1.000, n=5); ceiling 1000/b = 357 tok/s
- fit transformer (decode-tf-65536.json): step_ms = 7.52 + 2.976 * N (r2 0.999, n=3); ceiling 1000/b = 336 tok/s

### context 126,976 tokens

| family | N | step ms (median) | aggregate tok/s | per-seq tok/s | KV GiB | pool GiB | peak GiB |
|---|---|---|---|---|---|---|---|
| transformer | 1 | 13.17 | 75.9 | 75.94 | 15.51 | 15.5 | 45.4 |
| transformer | 2 | 19.08 | 104.8 | 52.40 | 31.02 | 31.0 | 61.9 |
| transformer | 3 | 24.53 | 122.3 | 40.77 | 46.52 | 46.5 | 64.2 |
| transformer | 4 | OOM | | | | | |
| lastkv | 1 | 13.82 | 72.4 | 72.37 | 0.73 | 0.7 | 17.2 |
| lastkv | 2 | 19.82 | 100.9 | 50.45 | 1.47 | 1.5 | 18.0 |
| lastkv | 4 | 30.74 | 130.1 | 32.53 | 2.94 | 2.9 | 19.7 |
| lastkv | 8 | 52.39 | 152.7 | 19.09 | 5.88 | 5.9 | 23.1 |
| lastkv | 16 | 95.71 | 167.2 | 10.45 | 11.75 | 11.8 | 29.9 |
| lastkv | 32 | 181.84 | 176.0 | 5.50 | 23.50 | 23.5 | 43.5 |
| lastkv | 48 | 274.91 | 174.6 | 3.64 | 35.25 | 35.3 | 57.1 |
| lastkv | 64 | 356.44 | 179.6 | 2.81 | 47.00 | 47.0 | 70.6 |
| lastkv | 72 | OOM | | | | | |

- fit lastkv (decode-lastkv-126976-a.json): step_ms = 8.77 + 5.440 * N (r2 1.000, n=5); ceiling 1000/b = 184 tok/s
- fit lastkv (decode-lastkv-126976-b.json): step_ms = 9.17 + 5.456 * N (r2 0.999, n=3); ceiling 1000/b = 183 tok/s
- fit transformer (decode-tf-126976.json): step_ms = 7.25 + 5.915 * N (r2 1.000, n=2); ceiling 1000/b = 169 tok/s

## FlashInfer decode plan() host cost (no model; 32 dummy layers)

context 126,976 tokens, page 16, 32/8 heads x 128

| N | plan ms, CPU indptr + device indices (idle / busy) | plan ms, all inputs on device (idle / busy) | 32-layer step wall ms | step GPU ms |
|---|---|---|---|---|
| 1 | 0.057 / 0.064 | 0.070 / 0.078 | 5.93 | 5.90 |
| 4 | 0.067 / 0.073 | 0.081 / 0.090 | 22.11 | 22.09 |
| 16 | 0.097 / 0.102 | 0.110 / 0.123 | 86.76 | 86.70 |
| 41 | 0.147 / 0.162 | 0.162 / 0.183 | 242.50 | 242.43 |
| 64 | 0.201 / 0.221 | 0.217 / 0.239 | 347.48 | 347.37 |

## Teacher-forced parity versus the HF reference

| file | family | tokens | prefill tokens | mean abs dlogprob | p99 | top-1 | prefill path | decode path | greedy |
|---|---|---|---|---|---|---|---|---|---|
| parity-lastkv-5120-split2496.json | lastkv | 5120 | 2496 | 0.0176 | 0.126 | 0.9926 | 0.0163 | 0.0188 | -/- |
| parity-lastkv-5120-split2500-fix.json | lastkv | 5120 | 2500 | 0.0179 | 0.128 | 0.9922 | 0.0163 | 0.0194 | -/- |
| parity-lastkv-5120-split2500.json | lastkv | 5120 | 2500 | 0.0260 | 0.232 | 0.9889 | 0.0163 | 0.0352 | -/- |
| parity-lastkv-5120-split2504-fix.json | lastkv | 5120 | 2504 | 0.0171 | 0.118 | 0.9912 | 0.0164 | 0.0178 | -/- |
| parity-lastkv-5120-split2504.json | lastkv | 5120 | 2504 | 0.0171 | 0.118 | 0.9912 | 0.0164 | 0.0178 | -/- |
| parity-lastkv-5120-split3072.json | lastkv | 5120 | 3072 | 0.0176 | 0.133 | 0.9918 | 0.0161 | 0.0199 | -/- |
| parity-lastkv-5120-split4000.json | lastkv | 5120 | 4000 | 0.0175 | 0.130 | 0.9912 | 0.0171 | 0.0188 | 32/32 |
| parity-tf-5120-split2496.json | transformer | 5120 | 2496 | 0.0208 | 0.120 | 0.9879 | 0.0211 | 0.0204 | -/- |
| parity-tf-5120-split2500-fix.json | transformer | 5120 | 2500 | 0.0207 | 0.120 | 0.9871 | 0.0211 | 0.0202 | -/- |
| parity-tf-5120-split2500.json | transformer | 5120 | 2500 | nan | nan | 0.8728 | 0.0211 | nan | -/- |
| parity-tf-5120-split2504-fix.json | transformer | 5120 | 2504 | 0.0206 | 0.119 | 0.9881 | 0.0211 | 0.0201 | -/- |
| parity-tf-5120-split2504.json | transformer | 5120 | 2504 | nan | nan | 0.9262 | 0.0211 | nan | -/- |
| parity-tf-5120-split4000.json | transformer | 5120 | 4000 | 0.0211 | 0.121 | 0.9871 | 0.0214 | 0.0198 | 10/32 |

## Prefill/decode hand-off: one request's state, GPU to GPU

peer access: True

| tokens | LastKV GB | Transformer GB | ratio | LastKV ms | Transformer ms | GB/s |
|---|---|---|---|---|---|---|
| 24,576 | 0.369 | 3.221 | 8.7 | 0.94 | 8.1 | 391.2 |
| 32,768 | 0.403 | 4.295 | 10.7 | 1.03 | 10.79 | 391.8 |
| 65,536 | 0.537 | 8.590 | 16.0 | 1.36 | 21.68 | 393.8 |
| 126,976 | 0.789 | 16.643 | 21.1 | 2.0 | 42.11 | 394.2 |


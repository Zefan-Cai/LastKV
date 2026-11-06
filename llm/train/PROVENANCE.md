# Provenance of the vendored trainer

`flame/` is a vendored copy of [flame](https://github.com/fla-org/flame),
Copyright (c) 2023-2025 Songlin Yang and Yu Zhang, MIT.  Its license is
reproduced in the repository's `LICENSES/flame-LICENSE`.

One trainer serves 124M, 760M and 8B.  This file records where each file came
from and every change, so the changes can be re-derived or sent upstream.

## Sources

| Snapshot | Revision | Contributes |
| --- | --- | --- |
| upstream flame | `24dea91` | the trainer this fork descends from |
| `Zefan-Cai/llm` | `4024dd9e7` (`main`) | `flame/` as vendored here; the 124M / 760M models and configs |
| `Zefan-Cai/llm` | `580a6c2a46b68a0312d5e7df8ae1d30f8734c812` | the 8B models and configs; `flame/components/optimizer.py` |

The second revision is the 256K source revision recorded in
`THIRD_PARTY_NOTICES.md`.  The 124M perlayerkv and sharekv model configs come
from a side branch of the same repository.

## Changes relative to upstream flame

Added modules:

| File | Purpose |
| --- | --- |
| `flame/checkpoint_manager.py` | checkpoint manager used instead of torchtitan's |
| `flame/eval_loss.py` | per-token-position loss evaluation, the PTL metric of Table 1 |
| `flame/utils/per_position_loss_tracker.py` | accumulator behind `eval_loss.py` |
| `flame/tools/loss_logging.py` | global loss metrics for logging |
| `flame/components/optimizer.py` | weight-decay masking, see below |

Modified modules:

| File | Change |
| --- | --- |
| `flame/train.py` | Weights & Biases login from `WANDB_API_KEY` / `WANDB_BASE_URL`, defaulting to the public host; barriers after dataset and checkpoint construction; parameter-init logging; LastKV loss metrics |
| `flame/data.py` | accept pre-tokenized `input_ids` and explicit `labels` / `attention_mask`, so the Arrow shards written by `data/prepare_long_data_collections.py` train without re-tokenization |
| `flame/config_manager.py` | options the added modules read, including `--optimizer.decay_embed_norm` |
| `flame/models/parallelize_fla.py` | sharding adjustments; the published runs use FSDP with `tensor_parallel_degree = 1`, so the tensor- and pipeline-parallel paths are unexercised |

## Weight decay

`flame/components/optimizer.py` comes from the 8B revision and is a superset of
the small-scale one.  Two independent masks:

- the LastKV blend gates (`prev_chunk_alpha(s)`, `first_chunk_alpha(s)`, and the
  8B `deep_v_scale` / `deep_k_scale`) are excluded from weight decay unless
  `--optimizer.alpha_weight_decay` is set.  Excluded in every published run.
- embeddings and RMSNorm gains are excluded unless
  `--optimizer.decay_embed_norm` is set.  The 8B run leaves this unset; the
  124M / 760M configs set it, because their recipe applies weight decay to
  everything except the blend gates.

Changing either flag changes the recipe, so the configs set them explicitly
rather than relying on defaults.

## Not included

Four trainer features of the 8B revision are deliberately left out.  Nothing in
`flame/` or `custom_models/` references them, and no shipped config sets their
options.

| Feature | What it did | Consequence |
| --- | --- | --- |
| Weight EMA (`ema_beta`) | marin-style weight averaging, decay 0.995, carried in checkpoints | the 8B run used it; training without it is a recipe change |
| Skip-step guard | skipped optimizer steps on gradient-norm or loss spikes | removes a stability guard; gradient clipping still applies |
| Activation statistics logger | per-module activation norms | metrics only |
| Benchmark-loss logger | periodic loss on held-out benchmark text | metrics only |

Also absent: the packed-document (`varlen`) path and the activation offloading
used by the 256K long-context stage.  This release therefore reproduces the 8K
continued-pretraining run of Table 6, not the subsequent 256K stage that
produced the `256k-step-001209` revision of `ZefanCai/LastKV-LLM-8B-Base`.

Features of the 8B revision that proved unnecessary for these models, and are
also absent: a rotary-buffer reinitialization helper (the 8B model recomputes
`inv_freq` itself in `_init_weights` and `post_init`), hookless activation
checkpointing (the model checkpoints its own layer chunks), and a step-1 loss
abort guard.

## Verification

Checked on one 8xH200 node with the trainer's own dependency set (PyTorch
2.8.0+cu128, Transformers 4.57.6, flash-linear-attention, torchtitan,
flash-attn 2.8.3, Python 3.10):

- all eight shipped `model_type` values register with the Transformers Auto
  classes, and `flame.train` imports;
- the three KV constructions at both small backbones, the three ablation model
  types, and both 8B families each run a bfloat16 forward and backward with
  finite logits and nonzero gradients;
- the weight-decay masks resolve as intended: with `decay_embed_norm = true`
  only the blend gates are excluded, and with the flag unset the embeddings and
  RMSNorm gains are excluded as well;
- a two-GPU FSDP run of `scripts/train_124m.sh`'s config completes five
  optimizer steps on pre-tokenized Arrow shards, saves a distributed
  checkpoint, and `flame.utils.convert_dcp_to_hf` exports it to a Hugging Face
  directory that `AutoModelForCausalLM.from_pretrained` reloads and decodes
  from.

Not checked: a full-scale run at any of the three scales, and multi-node
rendezvous.  `flash-linear-attention`'s fused kernels require CUDA, so the
model code does not run on CPU.

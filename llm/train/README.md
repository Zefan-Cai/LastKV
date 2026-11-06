# LastKV language-model training

Training code for the LastKV language models at three scales: 124M and 760M
from-scratch long-context pretraining, and 8B continued pretraining.

Inference lives one level up: `llm/` for the Transformers reference
implementation and `vllm/` for vLLM serving.

## Layout

One trainer serves all three scales, vendored here so nothing external has to
be cloned or patched.

```
flame/            the trainer (see PROVENANCE.md for its sources and changes)
custom_models/    LastKV model implementations, imported by flame/train.py
configs/124M      model JSONs + training TOML
configs/760M      model JSONs + training TOML
configs/8B        model JSONs + training TOML
scripts/          torchrun launchers
data/             corpus tokenization into Arrow shards
tests/            config and model-construction checks
```

## Paper results produced by this code

| Paper | Scale | Arm | Model config | Launch |
| --- | --- | --- | --- | --- |
| Table 1, A1, A2 (124M rows) | 124M | LastKV perlayerkv | `configs/124M/lastkv_124M_perlayerkv.json` | `ARM=perlayerkv scripts/train_124m.sh` |
| Table 1, A1, A2 (760M rows) | 760M | LastKV sharekv | `configs/760M/lastkv_760M_sharekv.json` | `ARM=sharekv scripts/train_760m.sh` |
| Table 1 Transformer rows | 124M / 760M | Transformer | `configs/*/transformer_*.json` | `ARM=transformer scripts/train_*.sh` |
| Table 9, A1 (KV construction) | 124M / 760M | reusekv, sharekv, perlayerkv | `configs/*/lastkv_*_{reusekv,sharekv,perlayerkv}.json` | `ARM=<arm> scripts/train_*.sh` |
| Table 13 (no recurrence, detached) | 124M / 760M | `..._lastnorecurrent`, `..._detach_lastkv` | set `model_type` in a copied config | `scripts/train_*.sh --model.config <your.json>` |
| Figure 7 (source-layer depth) | 124M | `..._source_layer` | set `model_type` and `source_layer` | `scripts/train_124m.sh --model.config <your.json>` |
| Table 6 | 8B | LastKV reusekv | `configs/8B/lastkv_8B_reusekv.json` | `scripts/train_8b.sh` |
| Table 6 Start / Transformer | 8B | Transformer | `configs/8B/transformer_8B.json` | `ARM=transformer scripts/train_8b.sh` |

The published checkpoint `ZefanCai/LastKV-LLM-8B-Base` is the 8B run at step
59,904 (`main`) and a subsequent 256K long-context stage at step 1,209
(`256k-step-001209`).  Table 6 evaluates step 18,432, which is not published.
The 124M and 760M checkpoints are not published.

This release reproduces the 8K continued-pretraining run.  The 256K stage needs
the packed-document path and the activation offloading listed as absent in
[`PROVENANCE.md`](PROVENANCE.md), and is out of scope here.

Baselines other than Transformer (Mamba-2, DeltaNet, Gated DeltaNet, Kimi Delta
Attention, SWA, NSA, MoBA, LaCT, HySparse) are not in this release; they are
upstream `fla` models configured as described in the paper's appendix.

## Install

Use an environment separate from the inference extras.

```bash
pip install -r requirements.txt
pip install --no-use-pep517 \
  'flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention@3c6f70d'
pip install 'torchtitan @ git+https://github.com/pytorch/torchtitan.git@0b44d4c'
pip install flash-attn --no-build-isolation
pip install liger-kernel          # 8B only
```

The trainer is already in `flame/` and resolves `import custom_models` to this
directory's `custom_models/`, so there is nothing to clone or patch.  The launch
scripts put the training root on `PYTHONPATH` for you.

## Data

The 124M / 760M runs use Together AI's Long-Data-Collections, which is gated on
Hugging Face.  Tokenize any text corpus into the Arrow shards flame reads:

```bash
python data/prepare_long_data_collections.py \
  --dataset togethercomputer/Long-Data-Collections \
  --dataset-name default --split train \
  --out /data/lastkv/ldc-arrow --num-shards 256 --num-proc 32

export LASTKV_DATA_FILES='/data/lastkv/ldc-arrow/data-*.arrow'
```

Each document gets one leading BOS token and no EOS; documents are shuffled at
the document level with seed 42.  The trainer concatenates and cuts the stream
into 32,768-token sequences online, without document-boundary masking, so a
sequence may span several documents and every position is supervised.

The configs name the tokenizer by its Hugging Face id, which needs network
access on the training node.  On an offline cluster, download it once and pass
the directory instead:

```bash
ARM=perlayerkv scripts/train_124m.sh --model.tokenizer_path /data/transformer-1.3B-100B
```

The 8B continued-pretraining corpus is described in the paper's data appendix.
Its mixture is built from public Hugging Face datasets, but the builder scripts
named there are not part of this release.  Point `LASTKV_DATA_FILES` at your own
Parquet shards with a `text` column.

## Train

124M, one node of eight GPUs, per-GPU batch 4 for a global batch of 32
sequences, 10,240 updates:

```bash
export LASTKV_DATA_FILES='/data/lastkv/ldc-arrow/data-*.arrow'
ARM=perlayerkv NGPU=8 scripts/train_124m.sh
```

760M, four nodes of eight GPUs, per-GPU batch 1, 40,960 updates:

```bash
NNODE=4 NGPU=8 NODE_RANK=$RANK MASTER_ADDR=$MASTER_ADDR \
  ARM=sharekv scripts/train_760m.sh
```

8B continued pretraining:

```bash
export LASTKV_DATA_FILES='/data/cpt/*.parquet'
export LASTKV_INIT_CKPT=/data/ckpt/base-dcp
NNODE=56 NGPU=8 scripts/train_8b.sh
```

Any flag after the script name is forwarded to `flame.train`, so
`--training.steps`, `--optimizer.lr` and friends work as usual.

Optional Weights & Biases logging reads `WANDB_API_KEY` and, for a self-hosted
instance, `WANDB_BASE_URL`.  No credentials are stored in this repository.

## Hyperparameters

The TOMLs carry the published settings: AdamW with peak learning rate 1e-3,
betas 0.9 / 0.95, eps 1e-15, 1,024 linear warmup updates then cosine decay to a
tenth of peak, gradient-norm clipping at 1.0, no gradient accumulation, seed 42.
Weight decay is 0.1 and is not applied to the LastKV blend gates, which is what
`flame/components/optimizer.py` in the patch implements.

The 8B run departs from this: peak learning rate 1.7e-3 held constant after a
re-warmup, weight decay 0.05, eps 1e-8, and gradient clipping 0.5.  It also
masks weight decay off embeddings and RMSNorm gains, which the small scales do
not; both behaviours come from the same optimizer and are selected by
`decay_embed_norm` in the config.  See `configs/8B/cpt_8B_8k.toml`.

The 8B run as executed also used weight EMA with decay 0.995 and a step-skip
guard.  Neither is part of this release; see [`PROVENANCE.md`](PROVENANCE.md).

## Checkpoints

flame writes distributed checkpoints.  Convert one to the Hugging Face layout
that `llm/` and `vllm/` load:

```bash
python -m flame.utils.convert_dcp_to_hf \
  --path exp/lastkv_8B_8k/checkpoint/step-59904 \
  --output /data/export/lastkv-8b --config configs/8B/lastkv_8B_reusekv.json
```

The 124M and 760M models use a different config class than the 8B family, so
their exports are not loadable by `lastkv.llm`; they load through
`AutoModelForCausalLM` once `models/` is importable.

## Tests

```bash
python -m pytest tests
```

`tests/test_configs.py` checks the configs against the paper's backbone and
recurrence settings and fails if an internal absolute path survived packaging.
`tests/test_models_import.py` builds a tiny instance of each KV construction and
runs one forward pass; it skips when `fla` is not installed.

## Provenance

See [`PROVENANCE.md`](PROVENANCE.md) for the source revisions, every change
relative to upstream flame, and the four trainer features this release leaves
out.  Launch scripts, data preparation, and tests in this directory are new.

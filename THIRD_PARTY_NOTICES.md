# Third-party notices

The repository's MIT license covers LastKV-owned code, examples, and documentation.
The following components retain their original licenses and copyright notices.

## Language model

`src/lastkv/llm/` packages the LastKV recurrent Marin implementation from
`Zefan-Cai/llm`, using the 8K source revision
`3d914d3217c2f3ea1513852e621c146099ece394` and the 256K source revision
`580a6c2a46b68a0312d5e7df8ae1d30f8734c812`.

That research code derives from [flame](https://github.com/fla-org/flame),
Copyright (c) 2023-2025 Songlin Yang and Yu Zhang, licensed under MIT.
Its license is reproduced in [LICENSES/flame-LICENSE](LICENSES/flame-LICENSE).
The release imports Llama configuration, rotary embeddings, and related utilities
from Hugging Face Transformers, which is distributed under Apache-2.0.

## Wan video model

The Wan-derived files in `src/lastkv/wan/` retain Apache-2.0:
`attention.py`, `layers.py`, `model.py`, `pipeline.py`, `scheduler.py`, `t5.py`,
`tokenizers.py`, and `vae.py`.

These components derive from [Wan2.1](https://github.com/Wan-Video/Wan2.1),
Copyright 2024-2025 The Alibaba Wan Team Authors. The LastKV adaptation is
packaged from research revision `b65c563ffac350811444cab1710c6903a6295215`
of `Zefan-Cai/world_models`. Changes include the LastKV causal attention/cache
path, portable PyTorch attention, standalone model loading, and inference-only
packaging. See [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

The strict-loading and shared-cache helpers (`loading.py`, `cache.py`) and
package entry point are LastKV-owned MIT code.

## Novel-view synthesis

`src/lastkv/nvs/` is an inference-only port of the LastKV research implementation
in `Zefan-Cai/ttt_lvsm`, with camera preprocessing and rendering examples. It is
released under the repository's MIT license. Checkpoint compatibility and source
configuration evidence are described in [docs/nvs.md](docs/nvs.md).

## Dependencies and model assets

PyTorch, Transformers, Diffusers, and the other installed dependencies retain
their own licenses. They are dependencies rather than bundled source trees.
Model checkpoints, VAE/text-encoder weights, and tokenizers are downloaded
separately and retain the licenses in their source repositories.

"""Tokenize a long-context corpus into Arrow shards for flame.

Reproduces the data pipeline of the 124M / 760M runs: one leading BOS token per
document and no EOS, then a document-level shuffle with seed 42.  flame
concatenates and cuts the stream into fixed ``training.seq_len`` sequences
online, without document-boundary masking, so every position is supervised.

The paper's corpus is Together AI's Long-Data-Collections, which is gated on
Hugging Face.  ``--dataset`` accepts any text dataset with a ``text`` or
``content`` column.

Example::

    python data/prepare_long_data_collections.py \
        --dataset togethercomputer/Long-Data-Collections \
        --dataset-name default --split train \
        --out /data/lastkv/ldc-arrow --num-shards 256 --num-proc 32

    export LASTKV_DATA_FILES='/data/lastkv/ldc-arrow/data-*.arrow'
"""

import argparse
import os

import pyarrow as pa
from datasets import load_dataset
from transformers import AutoTokenizer

TOKENIZER = "fla-hub/transformer-1.3B-100B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--num-shards", type=int, default=256)
    ap.add_argument("--num-proc", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42, help="document-level shuffle seed")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.bos_token_id is None:
        raise ValueError(f"{args.tokenizer} has no BOS token; flame needs one per document.")

    dataset = load_dataset(
        path=args.dataset, name=args.dataset_name, split=args.split,
        data_dir=args.data_dir, num_proc=args.num_proc,
    )

    columns = dataset.column_names
    text_column = "text" if "text" in columns else "content"
    if text_column not in columns:
        raise ValueError(f"No 'text' or 'content' column in {columns}")

    def tokenize(batch):
        # add_special_tokens=True yields exactly one leading BOS and no EOS for
        # this tokenizer.
        return {"input_ids": tokenizer(batch[text_column], add_special_tokens=True)["input_ids"]}

    dataset = dataset.map(
        tokenize, batched=True, num_proc=args.num_proc,
        remove_columns=columns, desc="tokenizing",
    )
    dataset = dataset.shuffle(seed=args.seed)

    os.makedirs(args.out, exist_ok=True)
    for i in range(args.num_shards):
        shard = dataset.shard(num_shards=args.num_shards, index=i, contiguous=True)
        path = os.path.join(args.out, f"data-{i:05d}-of-{args.num_shards:05d}.arrow")
        table = shard.with_format("arrow").data.table
        with pa.OSFile(path, "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)

    print(f"wrote {args.num_shards} shards to {args.out}")
    print(f"export LASTKV_DATA_FILES='{os.path.join(args.out, 'data-*.arrow')}'")


if __name__ == "__main__":
    main()

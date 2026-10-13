"""Generate text with a released LastKV 8B checkpoint."""

import argparse

import torch
from transformers import AutoTokenizer

from lastkv.llm import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ZefanCai/LastKV-LLM-8B-Base", help="HF repository or local checkpoint directory")
    parser.add_argument("--revision", default="main", help="main (8K) or 256k-step-001209")
    parser.add_argument("--prompt", default="The key to understanding the universe is")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--attention-backend", choices=("sdpa", "eager"), default="sdpa")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    dtype = getattr(torch, args.dtype) if args.dtype else (torch.bfloat16 if args.device.startswith("cuda") else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, fix_mistral_regex=False,
    )
    model = load_model(args.model, revision=args.revision, device=args.device,
                       dtype=dtype, attention_backend=args.attention_backend)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        result = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                do_sample=False, use_cache=True,
                                pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(result[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a video with LastKV Wan EMA or a selected raw checkpoint revision."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default="ZefanCai/LastKV-Wan-1.3B")
    parser.add_argument("--revision", default="main", help="EMA main or the published step-*-raw branch")
    parser.add_argument("--base-model", help="Public Wan base repository or local directory with VAE and UMT5 assets")
    parser.add_argument("--output", default="outputs/lastkv-wan.mp4")
    parser.add_argument("--negative-prompt", default="deformed, low quality, bad aesthetics, poor aesthetics")
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from lastkv.videogen import LastKVWanPipeline
    from lastkv.videogen.pipeline import save_video

    pipeline = LastKVWanPipeline.from_pretrained(
        args.model, revision=args.revision, base_model=args.base_model, device=args.device)
    video = pipeline.generate(args.prompt, negative_prompt=args.negative_prompt, num_frames=args.frames,
                              height=args.height, width=args.width, steps=args.steps,
                              guidance_scale=args.guidance_scale, shift=args.shift, seed=args.seed)
    save_video(video, args.output, fps=args.fps)
    print(f"Saved {len(video)} frames to {args.output}")


if __name__ == "__main__":
    main()

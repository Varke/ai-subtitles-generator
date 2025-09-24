#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from autosubs.pipeline import AutosubsPipeline, build_arg_parser

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    pipe = AutosubsPipeline(
        model=args.model,
        device=args.device,
        compute=args.compute,
        font=args.font,
        piper_voice=args.piper_voice,
        piper_voice_config=args.piper_voice_config,
        piper_length_scale=args.piper_length_scale,
        keep_original_audio=args.keep_original_audio,
        subs_format=args.format,
    )
    pipe.run(args.input, args.output, args.lang)

if __name__ == "__main__":
    main()

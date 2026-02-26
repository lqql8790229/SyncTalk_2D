"""Command-line interface for SyncTalk."""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def cmd_preprocess(args):
    from .data.preprocessing import DataPreprocessor
    from .utils.device import get_device
    preprocessor = DataPreprocessor(fps=25)
    device = get_device(args.device)
    preprocessor.process(args.video_path, device=device)


def cmd_train(args):
    from .configs.base import SyncTalkConfig
    from .training.trainer import Trainer

    if args.config:
        config = SyncTalkConfig.load(args.config)
    else:
        config = SyncTalkConfig.from_resolution(args.resolution, asr_mode=args.asr)

    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.lr = args.lr
    config.train.dataset_dir = args.dataset_dir
    config.train.save_dir = args.save_dir
    config.train.use_syncnet = args.use_syncnet

    trainer = Trainer(config, device_str=args.device)

    if args.stage == "syncnet":
        trainer.train_syncnet(args.dataset_dir, args.syncnet_save_dir)
    elif args.stage == "unet":
        trainer.train_unet(
            args.dataset_dir, args.save_dir,
            syncnet_checkpoint=args.syncnet_checkpoint,
            resume_from=args.resume,
        )
    elif args.stage == "full":
        trainer.train_full(args.dataset_dir, args.save_dir, args.syncnet_save_dir)


def cmd_inference(args):
    from .configs.base import SyncTalkConfig
    from .inference.engine import InferenceEngine

    if args.config:
        config = SyncTalkConfig.load(args.config)
    else:
        config = SyncTalkConfig.from_resolution(args.resolution, asr_mode=args.asr)

    engine = InferenceEngine(config, device_str=args.device)
    engine.generate(
        name=args.name,
        audio_path=args.audio_path,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        start_frame=args.start_frame,
        use_parsing=args.parsing,
    )


def cmd_serve(args):
    import uvicorn
    uvicorn.run(
        "synctalk.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def main():
    parser = argparse.ArgumentParser(
        description="SyncTalk - Commercial-grade 2D lip-sync video generation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Preprocess
    p_prep = subparsers.add_parser("preprocess", help="Preprocess training video")
    p_prep.add_argument("video_path", help="Path to input video")
    p_prep.add_argument("--device", default="auto")

    # Train
    p_train = subparsers.add_parser("train", help="Train models")
    p_train.add_argument("--stage", choices=["syncnet", "unet", "full"], default="full")
    p_train.add_argument("--dataset_dir", required=True)
    p_train.add_argument("--save_dir", default="./checkpoint/default")
    p_train.add_argument("--syncnet_save_dir", default="./syncnet_ckpt/default")
    p_train.add_argument("--syncnet_checkpoint", default=None)
    p_train.add_argument("--resolution", type=int, default=328)
    p_train.add_argument("--asr", default="ave")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch_size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--use_syncnet", action="store_true")
    p_train.add_argument("--resume", default=None, help="Resume from checkpoint")
    p_train.add_argument("--config", default=None, help="YAML config file")
    p_train.add_argument("--device", default="auto")

    # Inference
    p_infer = subparsers.add_parser("inference", help="Generate lip-sync video")
    p_infer.add_argument("--name", required=True, help="Dataset name")
    p_infer.add_argument("--audio_path", required=True, help="Audio WAV path")
    p_infer.add_argument("--checkpoint", default=None)
    p_infer.add_argument("--output", default=None)
    p_infer.add_argument("--resolution", type=int, default=328)
    p_infer.add_argument("--asr", default="ave")
    p_infer.add_argument("--start_frame", type=int, default=0)
    p_infer.add_argument("--parsing", action="store_true")
    p_infer.add_argument("--config", default=None, help="YAML config file")
    p_infer.add_argument("--device", default="auto")

    # Serve
    p_serve = subparsers.add_parser("serve", help="Start API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "inference": cmd_inference,
        "serve": cmd_serve,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

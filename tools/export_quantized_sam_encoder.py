import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export quantized SAM image encoder to a TorchScript .pb file"
    )
    parser.add_argument(
        "--predictor-pth",
        required=True,
        help="Path to torch.save()'d quantized SAM predictor (.pth)",
    )
    parser.add_argument(
        "--out-pb",
        required=True,
        help="Output .pb path for TorchScript image encoder",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to load/export on (cpu or cuda)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Input image size used by SAM image encoder",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    predictor = torch.load(args.predictor_pth, map_location=device)
    if not hasattr(predictor, "model") or not hasattr(predictor.model, "image_encoder"):
        raise RuntimeError("Loaded predictor does not contain model.image_encoder")

    image_encoder = predictor.model.image_encoder
    image_encoder.eval()
    image_encoder.to(device)

    dummy = torch.randn(1, 3, args.image_size, args.image_size, device=device)

    try:
        scripted = torch.jit.script(image_encoder)
    except Exception:
        scripted = torch.jit.trace(image_encoder, dummy)

    torch.jit.save(scripted, args.out_pb)


if __name__ == "__main__":
    main()

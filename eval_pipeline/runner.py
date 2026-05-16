import argparse
from .models import get_model, load_with_lora
from .evaluator import Evaluator


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="facebook/map-anything")
    p.add_argument("--lora_path", type=str, default=None,
                   help="Path to a saved LoRA adapter directory (omit for baseline)")
    p.add_argument("--styled_root", type=str, default=None,
                   help="Root of TeleStyle output (e.g. /scratch/.../telestyle_output)")
    p.add_argument("--style_name", type=str, default=None,
                   help="Style subdirectory name (e.g. impressionism)")
    p.add_argument("--baseline_name", type=str, default="mapanything_eval")
    p.add_argument("--max_scenes", type=int, default=None)
    p.add_argument("--max_pts", type=int, default=50000)
    p.add_argument("--out_dir", type=str, default="evaluation_results")
    return p.parse_args()


def main():
    args = get_args()
    if args.lora_path:
        model, device = load_with_lora(args.lora_path, base_checkpoint=args.checkpoint)
    else:
        model, device = get_model(args.checkpoint)
    evaluator = Evaluator(model=model, device=device,
                          baseline_name=args.baseline_name, max_pts=args.max_pts,
                          out_dir=args.out_dir,
                          styled_root=args.styled_root, style_name=args.style_name)
    evaluator.run(args.data_dir, max_scenes=args.max_scenes)


if __name__ == "__main__":
    main()

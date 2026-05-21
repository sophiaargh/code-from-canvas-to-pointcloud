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
    p.add_argument("--style_names", nargs="+",
                   default=["engraving", "impressionism", "oil_painting", "watercolor"],
                   help="Style subdirectory names to sample from")
    p.add_argument("--n_styled", type=int, default=4,
                   help="Number of styled views per scene in mixed mode")
    p.add_argument("--mixed", action="store_true",
                   help="Mixed mode: n_styled styled views + rest original photos per scene")
    p.add_argument("--baseline_name", type=str, default="mapanything_eval")
    p.add_argument("--max_scenes", type=int, default=None)
    p.add_argument("--max_pts", type=int, default=50000)
    p.add_argument("--out_dir", type=str, default="evaluation_results")
    p.add_argument("--grayscale", action="store_true",
                   help="Convert all images to grayscale-RGB before model inference")
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
                          styled_root=args.styled_root, style_names=args.style_names,
                          mixed=args.mixed, grayscale=args.grayscale, n_styled=args.n_styled)
    evaluator.run(args.data_dir, max_scenes=args.max_scenes)


if __name__ == "__main__":
    main()

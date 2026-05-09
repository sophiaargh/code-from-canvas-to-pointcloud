import argparse
from .models import get_model
from .evaluator import Evaluator


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="facebook/map-anything")
    p.add_argument("--baseline_name", type=str, default="mapanything_eval")
    p.add_argument("--max_scenes", type=int, default=None)
    p.add_argument("--view_ids", type=int, nargs="+", default=[0,10,20,30,40])
    p.add_argument("--max_pts", type=int, default=50000)
    p.add_argument("--out_dir", type=str, default="evaluation_results")
    return p.parse_args()


def main():
    args = get_args()
    model, device = get_model(args.checkpoint)
    evaluator = Evaluator(model=model, device=device, view_ids=args.view_ids,
                          baseline_name=args.baseline_name, max_pts=args.max_pts,
                          out_dir=args.out_dir)
    evaluator.run(args.data_dir, max_scenes=args.max_scenes)


if __name__ == "__main__":
    main()

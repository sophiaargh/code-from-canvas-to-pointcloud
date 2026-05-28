import argparse
from .models import get_model
from .evaluator import Evaluator
from .additional_evals.evaluator_pcd import DTUEvaluator
from .additional_evals.evaluator_style_degradation import StyleDegradeEvaluator
from .additional_evals.evaluator_icp import ICPSaveEvaluator

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",            type=str,  default="facebook/map-anything")
    p.add_argument("--baseline_name",         type=str,  default="mapanything_eval")
    p.add_argument("--style",                 type=str,  default="photographs")
    p.add_argument("--max_scenes",            type=int,  default=None)
    p.add_argument("--max_pts",               type=int,  default=50000)
    p.add_argument("--out_dir",               type=str,  default="evaluation_results")
    p.add_argument("--modification",               type=str,  default=None)
    p.add_argument("--encoder_block_prefix",  type=str,  default=None)
    p.add_argument("--norm_num_blocks",       type=int,  default=None)
    p.add_argument("--data_dir",              type=str, default=None, help="For the main evaluation, the dataset is loaded from HF, but not for the additional evaluators.")
    p.add_argument("--norm_from_end",         action="store_true",
                   help="Hook the LAST norm_num_blocks layers instead of the first.")
    p.add_argument("--norm_affine",           action="store_true",
                   help="Add learnable gamma/beta to each norm hook (prep for AdaIN).")
    p.add_argument("--adapter_weights",       type=str,  default=None,
                   help="Path to the saved adapter .pth file.")
    p.add_argument("--number_stylized", type=int, default=None)
    
    return p.parse_args()


def main():
    args = get_args()
    print(args)
    model, device = get_model(
        args.checkpoint,
        encoder_block_prefix = args.encoder_block_prefix,
        norm_num_blocks      = args.norm_num_blocks,
        norm_from_end        = args.norm_from_end,
        norm_affine          = args.norm_affine,
        adapter_weights      = args.adapter_weights,
    )
    evaluator = Evaluator(
        model         = model,
        device        = device,
        baseline_name = args.baseline_name,
        style         = args.style,
        max_pts       = args.max_pts,
        out_dir       = args.out_dir,
        modification  = args.modification,
        max_scenes=args.max_scenes,
    )
    print(f"Evaluating: {args.baseline_name} on {args.max_scenes} scenes, output in: {args.out_dir}")


    # Additionnal evaluators: Pointclouds on sample DTU with visualization and same ICP alignment and stylization ratio
    # evaluator = DTUEvaluator(model=model, device=device,
    #                       baseline_name=args.baseline_name, max_pts=args.max_pts,
    #                       out_dir=args.out_dir)
    # evaluator = StyleDegradeEvaluator(model=model, device=device,
    #                       out_dir=args.out_dir, number_stylized = args.number_stylized)
    # evaluator = ICPSaveEvaluator(model=model, device=device,
    #                       baseline_name=args.baseline_name, max_pts=args.max_pts,
    #                       out_dir=args.out_dir, photo_transforms_dir="photo_transforms")
    
    evaluator.run()


if __name__ == "__main__":
    main()
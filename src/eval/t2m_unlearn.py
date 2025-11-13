from collections import defaultdict

import numpy as np
import wandb

from pathlib import Path
import torch

from src.momask_codes.options.eval_option import EvalT2MOptions
from src.momask_codes.utils.get_opt import get_opt
from src.momask_codes.motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from src.momask_codes.models.t2m_eval_wrapper import EvaluatorModelWrapper
from src.momask_codes.utils.fixseed import fixseed

from src.eval.eval_fn_t2m import eval_t2m_unlearn


def get_model_loaders(model_name):
    if 'bamm' in model_name:
        from src.bamm.models.loaders import load_res_model, load_trans_model, load_vq_model
        print("Using BAMM models")
    else:
        from src.momask_codes.models.loaders import load_res_model, load_trans_model, load_vq_model
        print("Using Momask models")
    return load_res_model, load_trans_model, load_vq_model


@torch.no_grad()
def main():
    parser = EvalT2MOptions()
    add_eval_args(parser.parser)
    opt = parser.parse()
    fixseed(opt.seed)

    load_res_model, load_trans_model, load_vq_model = get_model_loaders(opt.name)
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else f"cuda:{opt.gpu_id}")
    if '-w-' in opt.split:
        split_tag = 'forget'
    elif '-wo-' in opt.split:
        split_tag = 'retain'
    else:
        split_tag = 'full'
    wandb.init(
        config=opt,
        resume='allow'
    )

    # Set up paths using pathlib
    opt.checkpoints_dir = Path(opt.checkpoints_dir)
    checkpoints_path = opt.checkpoints_dir / opt.dataset_name

    root_dir = checkpoints_path / opt.name
    model_dir = root_dir / 'model'

    model_opt_path = root_dir / 'opt.txt'
    model_opt = get_opt(str(model_opt_path), device=opt.device)

    vq_opt_path = opt.checkpoints_dir / opt.dataset_name / model_opt.vq_name / 'opt.txt'
    vq_opt = get_opt(str(vq_opt_path), device=opt.device)
    vq_opt.dim_pose = 251 if opt.dataset_name == 'kit' else 263
    vq_model, vq_opt = load_vq_model(vq_opt, opt.ckpt, device=opt.device)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    mtrans_model = load_trans_model(model_opt, opt.ckpt, device=opt.device)

    res_opt_path = opt.checkpoints_dir / opt.dataset_name / opt.res_name / 'opt.txt'
    res_opt = get_opt(str(res_opt_path), device=opt.device)
    res_model = load_res_model(res_opt, opt.ckpt, vq_opt, device=opt.device)

    assert res_opt.vq_name == model_opt.vq_name

    dataset_opt_path = str(opt.checkpoints_dir / opt.dataset_name / 'Comp_v6_KLD005' / 'opt.txt')

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, opt.split, device=opt.device)

    mtrans_model.eval()
    vq_model.eval()
    res_model.eval()

    mtrans_model.to(opt.device)
    vq_model.to(opt.device)
    res_model.to(opt.device)

    metrics_list = defaultdict(list)

    for i in range(opt.repeat_times):
        print(f"Evaluation repeat {i+1}/{opt.repeat_times}...")
        metrics = eval_t2m_unlearn(
            eval_val_loader, 
            vq_model, res_model, mtrans_model,
            eval_wrapper=eval_wrapper,
            time_steps=opt.time_steps, cond_scale=opt.cond_scale,
            temperature=opt.temperature, topkr=opt.topkr,
            force_mask=opt.force_mask, cal_mm=not opt.no_multimodality, 
            toxic_terms=opt.toxic_terms)

        for key, value in metrics.items():
            metrics_list[key].append(value)
            wandb.log({f"{split_tag}/repeats/{key}": value})

    for key, values in metrics_list.items():
        mean_val = np.mean(values)
        conf_interval = np.std(values) * 1.96 / np.sqrt(opt.repeat_times)
        print(f"\t{key.upper()}: {mean_val:.3f}, conf. {conf_interval:.3f}\n")

        # Log results to wandb
        wandb.log(
            {f"{split_tag}/{key}": mean_val, f"{split_tag}/conf/{key}": conf_interval},
            commit=False,
        )



def add_eval_args(parser):
    parser.add_argument('--ckpt', type=str, default='net_best_fid.tar',
                        help='checkpoint file to load')
    parser.add_argument('--split', type=str, default='test',
                        help='split to evaluate on, default: test')
    parser.add_argument('--run_name', type=str, default='eval_run',
                        help='name of the evaluation run for logging purposes')
    parser.add_argument('--no_multimodality', action='store_true',
                        help='whether to calculate multimodality metrics')
    parser.add_argument('--toxic_terms', type=str, nargs='+', default=[],
                        help='list of toxic terms to mask in the text')
    parser.add_argument('--method', type=str, help='method used for unlearning')
    parser.add_argument('--model_name', type=str, default='momask', help='name of the model to use')

if __name__ == "__main__":
    main()

# python eval_t2m_trans.py --name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_vq --dataset_name t2m --gpu_id 3 --cond_scale 4 --time_steps 18 --temperature 1 --topkr 0.9 --gumbel_sample --ext cs4_ts18_tau1_topkr0.9_gs
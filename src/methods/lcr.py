import argparse
import os
from os.path import join as pjoin
import sys

import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import trange

from src.momask_codes.models.vq.model import RVQVAE
from src.momask_codes.models.loaders import load_vq_model
from src.momask_codes.motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from src.momask_codes.utils.get_opt import get_opt
from src.momask_codes.options.eval_option import EvalT2MOptions
from src.momask_codes.utils.fixseed import fixseed



MAX_NUM_CAPTIONS = 4  # max number of captions per motion
# NUM_LAYERS = 1#6  # number of residual layers in the VQ-VAE
CODEBOOK_SIZE = 512  # number of codebooks per layer
# args.seq_len = 49  # sequence length


"""
This script is used to prune the codes of the VQ-VAE model based on the keywords in the captions.
The idea is to remove the codes that are associated with the specified keywords.
    1. Collect the codes of the VQ-VAE model for each motion (will take long the first time)
    2. Define positive samples based on the input split file: --kw_split = train_val-w-kick.txt
    3. Count the frequency of each code in the positive and negative samples
    4. Select the top-k codes based on the frequency and the ratio of the frequency
    5. Prune the selected codes from the codebook by assigning them to a random code (--dest_code_idx)
    
The script will produced a state_dict of the pruned model in the checkpoints folder named
checkpoints/HumanML3D/rvq_nq6_dc512_nc512_noshare_qdp0.2/model/
    {args.ext}.tar
    
ex:
To prune 16 codes relative to kicks:
python prune_codes.py \
    --name rvq_nq6_dc512_nc512_noshare_qdp0.2 \
    --dataset_name HumanML3D \
    --gpu_id 0 \
    --ext cp16\
    --kw_split train_val-w-kick.txt \
    --code_prune 16
    
There are additional arguments that can be used to control the pruning process:
--code_prune: Number of codes to prune (4,8,16,...)
--layer_prune: which layers to prune (0,1,2,...) default is 0 meaning only the first layer
--alpha: Alpha value for pruning: higher value means more aggressive pruning (default 1)
--orth: Orthogonal pruning: remove components orthogonal to the unwanted direction

I observed that alpha=1, orth=False, layer_prune=0 works well for the kick keyword
"""

def add_lcr_args(parser):
    parser.add_argument('--ckpt', type=str, default='latest.tar', help='Checkpoint file to load')
    parser.add_argument('--censor_codes_files', nargs='+', type=str, default=None, help='List of yaml files with codes to censor')
    parser.add_argument('--codes_csv', type=str, default='../assets/codes/codes.csv', help='CSV file to save the collected codes')
    parser.add_argument('--dest_code_idx', type=int, default=0, help='Code index to assign the pruned codes to')
    parser.add_argument('--code_prune', type=int, default=16, help='Number of codes to prune')
    parser.add_argument('--layer_prune', type=int, default=1, help='Number of layers to prune')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha value for pruning')
    parser.add_argument('--orth', action='store_true', help='Use orthogonal pruning')
    parser.add_argument('--just_select_codes', action='store_true', help='Just select codes and exit without pruning')
    parser.add_argument('--max_num_captions', type=int, default=4, help='Maximum number of captions per motion')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of residual layers to apply the LCR model, 1 applies only to the base layer')
    parser.add_argument('--codebook_size', type=int, default=512, help='Size of the codebook')
    parser.add_argument('--seq_len', type=int, default=49, help='Sequence length for the motion data')
    parser.add_argument('--run_name', type=str, default='lcr_run', help='Name of the run for logging')
    parser.add_argument('--split', type=str, default='test',
                        help='Split used to select the codes, default: test')

@torch.no_grad()
def main():
    parser = EvalT2MOptions()
    add_lcr_args(parser.parser)
    args = parser.parse()
    fixseed(args.seed)
    args.device = torch.device("cpu" if args.gpu_id == -1 else f"cuda:{args.gpu_id}")

    print("Forget file: ", args.split)

    ##### ---- Load Network ---- #####
    vq_opt_path = pjoin(
        args.checkpoints_dir, args.dataset_name, args.vq_name, "opt.txt"
    )
    vq_opt = get_opt(vq_opt_path, device=args.device)
    net, _ = load_vq_model(
        vq_opt, args.ckpt, device=args.device
    )

    net.eval()
    net.to(args.device)

    ##### ---- Collect or Load Codes ---- #####
    if args.censor_codes_files:
        # Load codes from the specified yaml file
        censor_codes = [[] for _ in range(args.num_layers)]
        for file in args.censor_codes_files:
            codes_file = readin_codes(file)
            for layer in range(min(args.num_layers, len(codes_file))):
                censor_codes[layer].extend(codes_file[layer])
    else:
        if os.path.exists(args.codes_csv):
            # Get codes from the csv file
            df = pd.read_csv(args.codes_csv, dtype={col: str for col in range(3)})
            print(f"Loaded codes from {args.codes_csv}")
        else:
            # Collect codes from the dataset
            dataset_opt_path = pjoin(args.checkpoints_dir , args.dataset_name , 'Comp_v6_KLD005' , 'opt.txt')
            _, dataset = get_dataset_motion_loader(dataset_opt_path, args.batch_size, 'train_val', device=args.device)

            dfs = []
            print("Collecting codes...")
            for r in trange(args.repeat_times):
                # repeat the process to smooth out the randomness
                df = collect_codes(dataset, net, device=args.device)
                df["repeat"] = r
                dfs.append(df)
            df = pd.concat(dfs, ignore_index=True)
            df.to_csv(args.codes_csv, index=False)

        # ! Here kick stands for what we want to remove
        df["positive"] = define_positives(df, args.split)
        print(f'Positive samples: {df["positive"].sum() / len(df):.2%}')
        assert df["positive"].sum() > 0, "No positive samples found!"
        kick_codes = df[df["positive"]][
            [col for col in df.columns if col.startswith("codes")]
        ]
        rest_codes = df[~df["positive"]][
            [col for col in df.columns if col.startswith("codes")]
        ]
        print(f"Collected {len(kick_codes)} kick codes and {len(rest_codes)} rest codes.")

        kick_codes = [
            kick_codes[[f"codes_{l}:{i}" for i in range(args.seq_len)]].values
            for l in range(args.num_layers)
        ]
        print(kick_codes[0])
        rest_codes = [
            rest_codes[[f"codes_{l}:{i}" for i in range(args.seq_len)]].values
            for l in range(args.num_layers)
        ]
        print(f"Collected codes for {args.num_layers} layers.")

        kick_code_counts = [count_codes(kick_codes[l]) for l in range(args.num_layers)]
        rest_code_counts = [count_codes(rest_codes[l]) for l in range(args.num_layers)]
        print("Code counts collected.")

        ##### ---- Select Top Codes ---- #####
        censor_codes = [
            topk_codes(kick_code_counts[l], rest_code_counts[l], k=args.code_prune)
            for l in range(args.layer_prune)
        ]
        print(f"Selected censor codes: {censor_codes}")

    if args.just_select_codes:
        print(f"Selected codes written to assets/codes/{args.run_name}.yaml")
        # Just select codes and exit printing codes
        os.makedirs("assets/codes", exist_ok=True)
        writeout_codes(censor_codes, f'assets/codes/{args.run_name}.yaml')
        exit(0)

    ##### ---- Prune Codes ---- #####
    target_code_idx = args.dest_code_idx
    while target_code_idx in censor_codes[0]:
        print(f"Code {target_code_idx} is in the censor list. Selecting another code...")
        codebook_size = net.quantizer.codebooks.size(1)
        target_code_idx = np.random.randint(codebook_size)
    print(f"Target code: {target_code_idx}")
    
    target_code = net.quantizer.codebooks[:, target_code_idx].clone()
    std = net.quantizer.codebooks.std(dim=(1, 2))

    for i, codes in enumerate(censor_codes):
        quantizer = net.quantizer.layers[i]
        codebook = quantizer.codebook.clone()

        if args.orth:
            direction = codebook[codes].mean(dim=0)
            direction = direction / direction.norm()
            for code in codes:
                codebook[code] = (
                    codebook[code]
                    - codebook[code].dot(direction) * direction * args.alpha
                )
        else:
            codebook[codes] = target_code[i]
            codebook[codes] += torch.randn_like(codebook[codes]) * std[i] * args.alpha

        net.quantizer.layers[i].codebook = codebook

    save_path = pjoin(
        args.checkpoints_dir,
        args.dataset_name,
        args.vq_name,
        "model",
        f'{args.run_name}{"-orth"*args.orth}.tar',
    )
    torch.save(
        {
            "ep": args.which_epoch,
            "vq_model": net.state_dict(),
            "removed_codes": censor_codes,
            "alpha": args.alpha,
            "orth": args.orth,
            "layer_prune": args.layer_prune,
            "code_prune": args.code_prune,
            "split": args.split,
        },
        save_path,
    )
    print(f"Pruned model saved to {save_path}")


def readin_codes(yaml_file: str): # -> List[List[int]]:
    """Read in the codes from the yaml file"""
    with open(yaml_file, "r") as f:
        codes = yaml.safe_load(f)
    return codes

def writeout_codes(codes, yaml_file: str):
    """Write out the codes to the yaml file"""
    with open(yaml_file, "w") as f:
        yaml.safe_dump(codes, f, default_flow_style=False)
        

def count_codes(codes):
    print(f"Counting codes: {len(codes)}")
    
    a = [np.unique(c) for c in codes]
    a = np.concatenate(a)
    return np.bincount(a, minlength=CODEBOOK_SIZE)


def topk_codes(kick_codes, rest_codes, k=16):
    freqs = kick_codes / (rest_codes + 1e-8)  # avoid division by zero
    ratios = (kick_codes / (kick_codes.sum() + 1e-8)) / (rest_codes / (rest_codes.sum() + 1e-8) + 1e-8)  # avoid division by zero
    topk_freqs = np.argsort(-freqs)[:k]
    topk_ratios = np.argsort(-ratios)[:k]
    return np.unique(np.concatenate([topk_freqs, topk_ratios])).tolist()


def collect_codes(dataset, net: RVQVAE, device="cuda"):
    all_codes = [[] for _ in range(6)]
    all_texts = []
    all_names = []
    all_reps = []

    for i, data in enumerate(dataset):
        name = dataset.name_list[i]
        captions = dataset.data_dict[name]["text"]
        rep, name = name.split('_', 1)[-1].split('%') 
        all_names.append(
            name
        )  # remove extra underscore
        all_texts.append(captions[0]['caption'])
        all_reps.append(rep)

        motion = data[4]
        motion = torch.tensor(motion).to(device).unsqueeze(0)
        code_idx, _ = net.encode(motion)
        for l in range(6):
            all_codes[l].append(code_idx.squeeze()[:, l].cpu().numpy())

    # now compose everything into a pandas dataset
    all_codes = [np.vstack(codes) for codes in all_codes]

    df = pd.DataFrame(
        {
            "name": all_names,
            "rep": all_reps,
            "text": all_texts,
            **{
                f"codes_{l}:{i}": all_codes[l][:, i]
                for l in range(5)
                for i in range(49)
            },
        }
    )
    return df


def define_positives(df: pd.DataFrame, kw_split_file: str):
    """Define positive samples based on keywords in the captions"""
    with open(f"{kw_split_file}.txt", "r") as f:
        names = f.read().splitlines()
    df.index = df["name"]
    print(f"Defining positives from {kw_split_file}.txt with {len(names)} names.")
    print(f"Sample names: {names[:5]}")
    
    print("DataFrame names sample: ", df["name"].tolist()[:5])

    # At least one caption should contain the keyword
    positives = df["name"].isin(names)
    return positives


if __name__ == "__main__":  
    main()
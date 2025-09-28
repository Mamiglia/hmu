import numpy as np
from argparse import ArgumentParser
import wandb
import json
import pandas as pd

def parse_args():
    parser = ArgumentParser(description="Evaluate the detector on generated samples")
    parser.add_argument('--run_name', type=str, help='Experiment name extension for wandb logging')
    parser.add_argument('--file', type=str, required=True, help='Path to the json results file')
    parser.add_argument('--forget_kw', type=str, nargs='+', required=True, help='Keywords to forget')
    # parser.add_argument('--num_repetitions', type=int, default=1, help='Number of repetitions for evaluation')
    # parser.add_argument('--label2category_file', type=str, help='Path to label to category mapping file', default='/home/tao/hdd/momask-codes/dataset/HumanML3D/label2category.csv')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    wandb.init(
        resume='allow', )
    
    # Load the results file
    with open(args.file, 'r') as f:
        data = json.load(f)['records']
        
    for res in data:
        captions = [s['description'] for s in res['retrieval']]
        detected = np.array([any(kw in caption for kw in args.forget_kw) for caption in captions])
        
        if detected.any():
            rank = np.where(detected)[0][0] + 1
        else:
            rank = len(captions) + 1  # If no detection, consider it

        res['first_rank'] = rank
        
    names = [res['name'] for res in data]
    ranks = np.array([res['first_rank'] for res in data])
    rep = [res['repeat'] for res in data]

    df = pd.DataFrame({
        'name': names,
        'ranks': ranks,
        'repeat': rep
    })

    ncs_1, ncs_2, ncs_3, ncs_5, ncs_10, mean, median = [], [], [], [], [], [], []

    for r in df['repeat'].unique():
        # get minimum ranks for each name
        df_r = df[df['repeat'] == r].groupby('name').min().reset_index()       
     
        total = len(df_r)
        ncs_1.append((df_r.ranks <= 1).mean())
        ncs_2.append((df_r.ranks <= 2).mean())
        ncs_3.append((df_r.ranks <= 3).mean())
        ncs_5.append((df_r.ranks <= 5).mean())
        ncs_10.append((df_r.ranks <= 10).mean())
        mean.append(np.mean(df_r.ranks))
        median.append(np.median(df_r.ranks))
        
    def confidence_interval(data, confidence=0.95):
        n = len(data)
        se = np.std(data, ddof=1) / np.sqrt(n)
        h = se * 1.96
        return h

    wandb.log({
        "unlearn/forget_size": total,
        "unlearn/ncs@1": np.mean(ncs_1),
        "unlearn/ncs@2": np.mean(ncs_2),
        "unlearn/ncs@3": np.mean(ncs_3),
        "unlearn/ncs@5": np.mean(ncs_5),
        "unlearn/ncs@10": np.mean(ncs_10),
        "unlearn/mean_rank": np.mean(mean),
        "unlearn/median_rank": np.mean(median),
        "unlearn/conf/ncs@1": confidence_interval(ncs_1),
        "unlearn/conf/ncs@2": confidence_interval(ncs_2),
        "unlearn/conf/ncs@3": confidence_interval(ncs_3),
        "unlearn/conf/ncs@5": confidence_interval(ncs_5),
        "unlearn/conf/ncs@10": confidence_interval(ncs_10),
        "unlearn/conf/mean_rank": confidence_interval(mean),
        "unlearn/conf/median_rank": confidence_interval(median),
    })
    
    # Min NCS
    
    # Get min rank for each name across repetitions
    min_ranks = df.groupby('name').ranks.min().reset_index()
    wandb.log({
        "unlearn/min_rank": min_ranks['ranks'].mean(),
        "unlearn/min_median_rank": np.median(min_ranks['ranks']),
        "unlearn/min_ncs@1": (min_ranks.ranks <= 1).mean(),
        "unlearn/min_ncs@2": (min_ranks.ranks <= 2).mean(),
        "unlearn/min_ncs@3": (min_ranks.ranks <= 3).mean(),
        "unlearn/min_ncs@5": (min_ranks.ranks <= 5).mean(),
        "unlearn/min_ncs@10": (min_ranks.ranks <= 10).mean(),
    })

    wandb.finish()
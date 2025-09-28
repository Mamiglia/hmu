import os
from os.path import join as pjoin

import torch
import torch.nn.functional as F

from src.momask_codes.models.loaders import load_res_model, load_trans_model, load_vq_model, load_len_estimator
from src.momask_codes.motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from src.momask_codes.models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from src.momask_codes.models.vq.model import RVQVAE, LengthEstimator

from src.momask_codes.options.eval_option import EvalT2MOptions
from src.momask_codes.utils.get_opt import get_opt

from src.momask_codes.utils.fixseed import fixseed
from src.momask_codes.visualization.joints2bvh import Joint2BVHConvertor
from torch.distributions.categorical import Categorical

import wandb
import json

from src.momask_codes.utils.motion_process import recover_from_ric
from src.momask_codes.utils.plot_script import plot_3d_motion

from src.momask_codes.utils.paramUtil import t2m_kinematic_chain
from tqdm import tqdm

import numpy as np
clip_version = 'ViT-B/32'

def add_viz_args(parser):
    parser.add_argument('--ckpt', type=str, default='latest.tar', help='Checkpoint file to load')
    parser.add_argument('--skip_viz', action='store_true', help='Skip visualization')
    parser.add_argument('--ik_viz', action='store_true', help='Use IK for visualization')
    parser.add_argument('--run_name', type=str, default='t2m_gen', help='Name of the run for wandb logging')
    parser.add_argument('--split', type=str, default='test', help='Main split to use for generation')


if __name__ == '__main__':
    parser = EvalT2MOptions()
    add_viz_args(parser.parser)
    opt = parser.parse()
    fixseed(opt.seed)
    
    wandb.init(
        resume='allow'
    )

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))

    dim_pose = 251 if opt.dataset_name == 'kit' else 263

    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    result_dir = pjoin('./generation', opt.run_name)
    joints_dir = pjoin(result_dir, 'joints')
    animation_dir = pjoin(result_dir, 'animations')
    feats_dir = pjoin(result_dir, 'feats')
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(animation_dir,exist_ok=True)
    os.makedirs(feats_dir, exist_ok=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)


    #######################
    ######Loading RVQ######
    #######################
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.dim_pose = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt, ckpt=opt.ckpt, device=opt.device)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    #################################
    ######Loading R-Transformer######
    #################################
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt, opt.ckpt, vq_opt, device=opt.device)

    assert res_opt.vq_name == model_opt.vq_name

    #################################
    ######Loading M-Transformer######
    #################################
    t2m_transformer = load_trans_model(model_opt, opt.ckpt, device=opt.device)

    ##################################
    #####Loading Length Predictor#####
    ##################################
    length_estimator = load_len_estimator(model_opt)

    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()
    length_estimator.eval()

    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)
    length_estimator.to(opt.device)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
    def inv_transform(data):
        return data * std + mean

    dataset_opt_path = pjoin(
        opt.checkpoints_dir, opt.dataset_name, "Comp_v6_KLD005", 'opt.txt'
    )
    
    dataloader, dataset = get_dataset_motion_loader(
        dataset_opt_path,
        opt.batch_size,
        opt.split,
        device=opt.device,
    )
    dataloader.dataset.return_names = True

    kinematic_chain = t2m_kinematic_chain
    converter = Joint2BVHConvertor()
    
    records = []

    for b_num,batch in enumerate(tqdm(dataloader)):
        (   word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            token,
            names
        ) = batch

        captions = caption
        token_lens = m_length // 4
        token_lens = token_lens.to(opt.device)
                        
        for r in range(opt.repeat_times):
            with torch.no_grad():
                mids = t2m_transformer.generate(captions, token_lens,
                                                timesteps=opt.time_steps,
                                                cond_scale=opt.cond_scale,
                                                temperature=opt.temperature,
                                                topk_filter_thres=opt.topkr,
                                                gsample=opt.gumbel_sample)
                # print(mids)
                # print(mids.shape)
                mids = res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
                pred_motions = vq_model.forward_decoder(mids)

                pred_motions = pred_motions.detach().cpu().numpy()

                data = inv_transform(pred_motions)
                
            for k, (caption, feats)  in enumerate(zip(captions, data)):
                clean_name = '%'.join(names[k].split('%')[1:])
                name = names[k] + "_" + '_'.join(caption.strip().split(' ')[-3:]) + f'_len{m_length[k]}'
                joint_path = pjoin(joints_dir,  str(r), f"{name}.npy")
                animation_path = pjoin(animation_dir,  str(r), f"{name}.mp4")
                feats_path = pjoin(feats_dir, str(r),  f"{name}.npy")

                os.makedirs(os.path.dirname(animation_path), exist_ok=True)
                os.makedirs(os.path.dirname(joint_path), exist_ok=True)
                os.makedirs(os.path.dirname(feats_path), exist_ok=True)

                feats = feats[:m_length[k]]
                joint = recover_from_ric(torch.from_numpy(feats).float(), 22).numpy()
                np.save(joint_path, joint)
                np.save(feats_path, feats)
                    
                records.append({
                    "caption": caption,
                    "name": clean_name,
                    "joint_path": joint_path,
                    "feats_path": feats_path,
                    # "bvh_path": bvh_path,
                    "video_path": animation_path if not opt.skip_viz else None,
                    'length': m_length[k].item(),
                    'batch': b_num,
                    'sample': k,
                    'repeat': r
                })
                
                if opt.skip_viz:
                    continue
                bvh_path = animation_path.replace('.mp4', '.bvh')
                _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)

                plot_3d_motion(animation_path, kinematic_chain, joint, title=caption, fps=20)


                wandb.log({
                    f'video/{name}': wandb.Video(animation_path, caption=caption), 
                    # f'video/{name}/caption':caption,
                    # f'video/{name}/name':name,
                    # f'video/{name}/length': m_length[k],
                    })

                if opt.ik_viz:
                    bvh_path = animation_path.replace('.mp4', '_ik.bvh')
                    _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)
                    ik_save_path = animation_path.replace('.mp4', '_ik.mp4')
                    plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=caption, fps=20)
                    np.save(joint_path.replace('.npy', '_ik.npy'), ik_joint)

    # make a json file to save the records with metadata

    records_path = pjoin(result_dir, "records.json")
    metadata = {
        "dataset_name": opt.dataset_name,
        "ckpt": opt.ckpt,
        "split": opt.split,
        "records": records
    }
    with open(records_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved records to {records_path}")
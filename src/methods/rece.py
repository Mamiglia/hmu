import argparse
import os
import sys
from copy import deepcopy
from os.path import join as pjoin
from typing import Optional

import numpy as np
import torch

from src.momask_codes.models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from src.momask_codes.models.mask_transformer.transformer_trainer import (
    MaskTransformerTrainer,
    ResidualTransformerTrainer,
)
from src.momask_codes.models.vq.model import RVQVAE, LengthEstimator
from src.momask_codes.models.loaders import load_vq_model, load_res_model, load_trans_model, load_len_estimator
from src.momask_codes.options.eval_option import EvalT2MOptions
from src.momask_codes.utils.fixseed import fixseed
from src.momask_codes.utils.get_opt import get_opt
from src.momask_codes.utils.motion_process import recover_from_ric
from src.momask_codes.utils.paramUtil import t2m_kinematic_chain
from src.momask_codes.utils.plot_script import plot_3d_motion
from src.momask_codes.visualization.joints2bvh import Joint2BVHConvertor

def get_model_loaders(model_name):
    if model_name == "bamm":
        from src.bamm.models.loaders import load_res_model, load_trans_model, load_vq_model
        print("Using BAMM models")
    else:
        from src.momask_codes.models.loaders import load_res_model, load_trans_model, load_vq_model
        print("Using Momask models")
    return load_res_model, load_trans_model, load_vq_model

clip_version = "ViT-B/32"

# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()


@torch.no_grad()
def close_form_emb_regzero(
    proj_layers,
    concept,
    device="cpu",
    with_to_k=True,
    save_path=None,
    save_name=None,
    regeular_scale=1e-3,
    seed=123,
):
    """Close form solution for the adversarial embedding.

    Args:
        proj_matrices: List of projection matrices
        concept: Concept embedding
    """
    proj_layers = [deepcopy(l) for l in proj_layers]

    # Eq. 8 in the paper
    mat1 = torch.eye(proj_layers[0].weight.shape[1]).to(device) * regeular_scale
    mat2 = torch.zeros(
        (proj_layers[0].weight.shape[1], proj_layers[0].weight.shape[1])
    ).to(device)

    for idx_, l in enumerate(proj_layers):
        mat1 = mat1 + torch.matmul(l.weight.T, l.weight)
        mat2 = mat2 + torch.matmul(l.weight.T, proj_layers[idx_].weight)
    coefficent = torch.matmul(torch.inverse(mat1), mat2)
    adv_embedding = torch.matmul(concept, coefficent.T).unsqueeze(0)

    return concept.unsqueeze(0), adv_embedding


@torch.no_grad()
def edit_model_adversarial(
    proj_layers,
    forget_emb,
    target_emb,
    retain_emb,
    layers_to_edit=None,
    lamb=0.1,
    erase_scale=0.1,
    preserve_scale=0.1,
    with_to_k=True,
    technique="tensor",
):
    """Edit the model adversarially.

    Args:
        proj_layers: List of projection matrices
        forget_emb: List of embeddings to forget
        target_emb: List of target embeddings
        retain_emb: List of embeddings to retain
    """
    ######################## START ERASING ###################################
    for layer_num in range(len(proj_layers)):
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue

        #### prepare input k* and v*
        mat1 = lamb * proj_layers[layer_num].weight
        mat2 = lamb * torch.eye(
            proj_layers[layer_num].weight.shape[1],
            device=proj_layers[layer_num].weight.device,
        )

        for cnt, (old_emb, new_emb) in enumerate(zip(forget_emb, target_emb)):
            context = old_emb.unsqueeze(0).detach()
            new_emb = new_emb.unsqueeze(0).detach()

            values = []
            for layer in proj_layers:
                if technique == "tensor":
                    o_embs = layer(old_emb).detach()
                    u = o_embs
                    u = u / u.norm()

                    new_embs = layer(new_emb).detach()
                    new_emb_proj = (u * new_embs).sum()

                    target = new_embs - (new_emb_proj) * u
                    values.append(target.detach())
                elif technique == "replace":
                    values.append(layer(new_emb).detach())
                else:
                    values.append(layer(new_emb).detach())

            context_vector = context.reshape(context.shape[0], context.shape[1], 1)
            context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
            value_vector = values[layer_num].reshape(
                values[layer_num].shape[0], values[layer_num].shape[1], 1
            )
            for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
            for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
            mat1 += erase_scale * for_mat1
            mat2 += erase_scale * for_mat2

        for old_emb, new_emb in zip(retain_emb, deepcopy(retain_emb)):
            context = old_emb.unsqueeze(0).detach()
            new_emb = new_emb.unsqueeze(0).detach()

            with torch.no_grad():
                values = [layer(new_emb[:]).detach() for layer in proj_layers]

            context_vector = context.reshape(context.shape[0], context.shape[1], 1)
            context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
            value_vector = values[layer_num].reshape(
                values[layer_num].shape[0], values[layer_num].shape[1], 1
            )
            for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
            for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
            mat1 += preserve_scale * for_mat1
            mat2 += preserve_scale * for_mat2

        # Update the projection matrix directly
        # with side effect on the original model
        proj_layers[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

    return proj_layers

def save_model_to_ckpt(
    mask_trans,
    res_trans,
    vq_vae,
    opt,
    name="unnamed.tar",
    mask_path: Optional[str] = None,
    res_path: Optional[str] = None,
    **_,
):

    t2m_trans_state_dict = mask_trans.state_dict()
    clip_weights = [
        e for e in t2m_trans_state_dict.keys() if e.startswith("clip_model.")
    ]
    for e in clip_weights:
        del t2m_trans_state_dict[e]
    state = {
        "t2m_transformer": t2m_trans_state_dict,
        # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
        # 'scheduler':self.scheduler.state_dict(),
        "ep": -1,
        # 'total_it': total_it,
    }
    out_dir = pjoin(mask_path, "model", name)
    torch.save(state, out_dir)
    print(f"Model saved to {out_dir}")

    res_trans_state_dict = res_trans.state_dict()
    clip_weights = [
        e for e in res_trans_state_dict.keys() if e.startswith("clip_model.")
    ]
    for e in clip_weights:
        del res_trans_state_dict[e]
    state = {
        "res_transformer": res_trans_state_dict,
        # 'opt_res_transformer': self.opt_res_transformer.state_dict(),
        # 'scheduler':self.scheduler.state_dict(),
        "ep": -1,
        # 'total_it': total_it,
    }

    out_dir = pjoin(res_path, "model", name)
    torch.save(state, out_dir)

    print(f"Model saved to {out_dir}")


def load(opt):
    load_res_model, load_trans_model, load_vq_model = get_model_loaders(opt.model_name)
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))

    dim_pose = 251 if opt.dataset_name == "kit" else 263

    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, "model")
    result_dir = pjoin("./generation", opt.ext)
    joints_dir = pjoin(result_dir, "joints")
    animation_dir = pjoin(result_dir, "animations")
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(animation_dir, exist_ok=True)

    model_opt_path = pjoin(root_dir, "opt.txt")
    model_opt = get_opt(model_opt_path, device=opt.device)

    #######################
    ######Loading RVQ######
    #######################
    vq_opt_path = pjoin(
        opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, "opt.txt"
    )
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.dim_pose = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt, ckpt=opt.ckpt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    #################################
    ######Loading R-Transformer######
    #################################
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, "opt.txt")
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt, opt.ckpt, vq_opt)

    assert res_opt.vq_name == model_opt.vq_name

    #################################
    ######Loading M-Transformer######
    #################################
    t2m_transformer = load_trans_model(model_opt, opt.ckpt)


    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()

    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == "kit" else 22

    kinematic_chain = t2m_kinematic_chain
    converter = Joint2BVHConvertor()
    return (
        t2m_transformer,
        res_model,
        vq_model,
        {
            # "inv_transform": inv_transform,
            # "token_lens": token_lens,
            "converter": converter,
            "kinematic_chain": kinematic_chain,
            # "m_length": m_length,
            "animation_dir": animation_dir,
            "opt": opt,
            "joints_dir": joints_dir,
            "vq_path": os.path.dirname(vq_opt_path),
            "res_path": os.path.dirname(res_opt_path),
            "mask_path": os.path.dirname(model_opt_path),
        },
    )



@torch.no_grad()
def forward(masked, res, vq, text, opt, hp):
    mids = masked.generate(
        text,
        hp["token_lens"],
        timesteps=opt.time_steps,
        cond_scale=opt.cond_scale,
        temperature=opt.temperature,
        topk_filter_thres=opt.topkr,
        gsample=opt.gumbel_sample,
    )
    # print(mids)
    # print(mids.shape)
    mids = res.generate(mids, text, hp["token_lens"], temperature=1, cond_scale=5)
    # print(mids)
    # print(mids.shape)
    pred_motions = vq.forward_decoder(mids)

    pred_motions = pred_motions.detach().cpu().numpy()

    data = hp["inv_transform"](pred_motions)
    return data


def main(forget_text, retain_text, target, opt, epochs=10, preserve_scale=0.1):
    ##### COPIED from test.py #####
    masked_transformer, res_transformer, vq_model, hp = load(opt)

    ##### END COPY #####
    proj_layers = [masked_transformer.cond_emb, res_transformer.cond_emb]

    concept_embedding = masked_transformer.encode_text(forget_text)
    retain_embedding = masked_transformer.encode_text(retain_text)
    target_embedding = masked_transformer.encode_text([target for _ in forget_text])
    print(concept_embedding.shape, target_embedding.shape, retain_embedding.shape)
    assert target_embedding.shape == concept_embedding.shape

    # UCE
    model = edit_model_adversarial(  # implements Eq. 3 in the paper (UCE)
        proj_layers, concept_embedding, target_embedding, retain_embedding, preserve_scale=preserve_scale
    )

    # Save UCE model checkpoint
    save_model_to_ckpt(
        masked_transformer, res_transformer, vq_model, name=f"UCE_{preserve_scale}.tar", **hp
    )

    # RECE
    for _ in range(epochs):  # 10 iterations
        adv_embedding = [
            close_form_emb_regzero(
                proj_layers, concept, device=masked_transformer.opt.device
            )[1].squeeze()
            for concept in concept_embedding
        ]

        adv_embedding = torch.stack(adv_embedding, dim=0)
        print(adv_embedding.shape, retain_embedding.shape)

        model = edit_model_adversarial(  # implements Eq. 3 in the paper (UCE)
            proj_layers, adv_embedding, target_embedding, retain_embedding, preserve_scale=preserve_scale
        )


    save_model_to_ckpt(
        masked_transformer,
        res_transformer,
        vq_model,
        name=f"RECE{epochs}_{preserve_scale}.tar",
        **hp,
    )

    return 0

if __name__ == "__main__":
    parser = EvalT2MOptions()
    parser.parser.add_argument('--preserve_scale', type=float, default=0.1)
    parser.parser.add_argument('--forget_text', type=str, nargs='+', required=True)
    parser.parser.add_argument('--retain_text', type=str, nargs='+', default=["walk", "run", "jump", "sit", "stand"])
    parser.parser.add_argument('--target_text', type=str, default="")
    parser.parser.add_argument('--epochs', type=int, default=3)
    parser.parser.add_argument('--ckpt', type=str, default="base.tar")
    parser.parser.add_argument('--model_name', type=str, default='momask', help='name of the model to use')
    opt = parser.parse()
    fixseed(opt.seed)

    main(
        forget_text=opt.forget_text,
        retain_text=opt.retain_text,
        target=opt.target_text,
        epochs=opt.epochs,
        opt=opt,
        preserve_scale=opt.preserve_scale,
    )

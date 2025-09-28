import torch
import numpy as np
import torch
import re

# from scipy import linalg
from src.momask_codes.utils.metrics import *

from src.momask_codes.utils.word_vectorizer import WordVectorizer

# from ..models.t2m_eval_wrapper import EvaluatorModelWrapper
# from ..models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
# from ..models.vq.model import RVQVAE

@torch.no_grad()
def eval_t2m_unlearn(
    val_loader,
    vq_model ,#: RVQVAE,
    res_model ,#: ResidualTransformer,
    trans ,#: MaskTransformer,
    eval_wrapper,# : EvaluatorModelWrapper,
    time_steps,
    cond_scale,
    temperature,
    topkr,
    toxic_terms,
    gsample=True,
    force_mask=False,
    cal_mm=True,
    res_cond_scale=5,
):
    trans.eval()
    vq_model.eval()
    res_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = np.zeros(3)
    R_precision = np.zeros(3)
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0
    # We want to compute the matching score with the text embedding having the toxic words masked
    matching_score_clean_real = 0
    matching_score_clean_pred = 0

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3
        
    toxic_distance = 0
    notox_accuracy = 0
    w_vectorizer = WordVectorizer("./glove", "our_vab")

    for i, batch in enumerate(val_loader):
        (
            word_embeddings,
            pos_one_hots,
            clip_text,
            sent_len,
            pose,
            m_length,
            token,
        ) = batch

        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
            # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(
                    clip_text,
                    m_length // 4,
                    time_steps,
                    cond_scale,
                    temperature=temperature,
                    topk_filter_thres=topkr,
                    gsample=gsample,
                    force_mask=force_mask,
                )

                # motion_codes = motion_codes.permute(0, 2, 1)
                # mids.unsqueeze_(-1)
                pred_ids = res_model.generate(
                    mids.cuda(),
                    clip_text,
                    m_length // 4,
                    temperature=1,
                    cond_scale=res_cond_scale,
                )
                # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)
                # pred_ids = torch.where(pred_ids==-1, 0, pred_ids)

                pred_motions = vq_model.forward_decoder(pred_ids)

                # pred_motions = vq_model.decoder(codes)
                # pred_motions = vq_model.forward_decoder(mids)

                em_pred = eval_wrapper.get_motion_embeddings(
                    pred_motions.clone(),
                    m_length,
                )
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(
                motion_multimodality_batch, dim=1
            )  # (bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(
                clip_text,
                m_length // 4,
                time_steps,
                cond_scale,
                temperature=temperature,
                topk_filter_thres=topkr,
                force_mask=force_mask,
            )

            # motion_codes = motion_codes.permute(0, 2, 1)
            # mids.unsqueeze_(-1)
            pred_ids = res_model.generate(
                mids.cuda(), clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale
            )
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)
            # pred_ids = torch.where(pred_ids == -1, 0, pred_ids)

            pred_motions = vq_model.forward_decoder(pred_ids)
            # pred_motions = vq_model.forward_decoder(mids)

            em_pred = eval_wrapper.get_motion_embeddings(
                pred_motions.clone(),
                m_length,
            )

        pose = pose.cuda().float()


        et, em = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pose, m_length
        )

        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(
            et.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
        )

        mm_dist = (et - em_pred).norm(dim=-1)
        temp_match = mm_dist.sum().item()
        R_precision += temp_R
        matching_score_pred += temp_match

        def mask_toxic_words(token, toxic_words, placeholder="***"):
            # Create a regex pattern that matches any toxic word
            pattern = re.compile("|".join(map(re.escape, toxic_words)), re.IGNORECASE)
            # Use list comprehension with regex substitution
            
            return [pattern.sub(placeholder, tok) for tok in token]

        def vectorize_tokens(w_vectorizer, tokens_l):
            w, p = [], []
            for tokens in tokens_l:
                word_embeddings, pos_one_hots = [], []
                for token in tokens.split("_"):
                    word_emb, pos_oh = w_vectorizer[token]
                    pos_one_hots.append(pos_oh[None, :])
                    word_embeddings.append(word_emb[None, :])
                w.append(word_embeddings)
                p.append(pos_one_hots)
            w = np.stack(w, axis=0).squeeze(2)
            p = np.stack(p, axis=0).squeeze(2)
            return torch.from_numpy(w), torch.from_numpy(p)

        token_clean = mask_toxic_words(list(token), toxic_terms)

        if any(t_clean != t for t_clean, t in zip(token_clean, token)):
            # print("Toxic words masked in the text")
            
            word_embeddings_clean, pos_one_hots_clean = vectorize_tokens(
                w_vectorizer, token_clean
            )
            et_clean = eval_wrapper.get_text_embeddings(
                word_embeddings_clean, pos_one_hots_clean, sent_len, m_length
            )
            
            mm_dist_clean = (et_clean - em_pred).norm(dim=-1)

            matching_score_clean_pred += mm_dist_clean.sum().item()
            
            toxic_distance += (mm_dist / mm_dist_clean).sum().item()
            notox_accuracy += (mm_dist_clean < mm_dist).sum().item()

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else nb_sample)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    matching_score_clean_real = matching_score_clean_real / nb_sample
    matching_score_clean_pred = matching_score_clean_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    print(
        f"--> \t Eva, FID. {fid:.4f}, "
        # f"Diversity Real. {diversity_real:.4f},
        f"Diversity. {diversity:.4f}, "
        f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, "
        f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f},"
        f"multimodality. {multimodality:.4f}",
        f"matching_score_clean_real. {matching_score_clean_real:.4f}, matching_score_clean_pred. {matching_score_clean_pred:.4f}",
        f"toxic_distance. {toxic_distance / nb_sample:.4f}, notox_accuracy. {notox_accuracy / nb_sample:.4f}",
    )
    return {
        "fid": fid,
        "diversity": diversity,
        "R_precision_1": R_precision[0],
        "R_precision_2": R_precision[1],
        "R_precision_3": R_precision[2],
        "MM-Dist": matching_score_pred,
        "multimodality": multimodality,
        "MM-Safe": matching_score_clean_pred,
        "toxic_distance": toxic_distance / nb_sample,
        "notox_accuracy": notox_accuracy / nb_sample,
    }

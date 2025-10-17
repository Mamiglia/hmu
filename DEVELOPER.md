# Developer Documentation

This document explains the internal architecture, code organization, and how different components interact within the Human Motion Unlearning (HMU) repository.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Code Organization](#code-organization)
- [Data Flow](#data-flow)
- [Component Interactions](#component-interactions)
- [Extending the Codebase](#extending-the-codebase)

---

## Architecture Overview

The HMU repository implements a pipeline for unlearning concepts from text-to-motion models. The architecture consists of:

```
┌─────────────────────────────────────────────────────────────┐
│                     HMU Pipeline                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Dataset Splitting                                        │
│     ├── Keyword-based filtering                             │
│     └── TMR-based refinement                                │
│                                                              │
│  2. Model Loading (from momask_codes)                        │
│     ├── VQ-VAE (motion tokenizer)                           │
│     ├── Mask Transformer (base motion generator)            │
│     └── Residual Transformer (motion refiner)               │
│                                                              │
│  3. Unlearning Methods (src/methods/)                        │
│     ├── LCR: Prune VQ-VAE codebook                          │
│     ├── UCE: Edit transformer embeddings                    │
│     ├── RECE: Iterative adversarial editing                 │
│     └── Fine-tuning: Retrain on retain set                  │
│                                                              │
│  4. Evaluation (src/eval/)                                   │
│     ├── Generate motions (gen_t2m_batch.py)                 │
│     ├── TMR retrieval (m2m_retrieval.py)                    │
│     ├── Compute metrics (eval_fn_t2m.py)                    │
│     └── NCS calculation (ncs_compute.py)                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Dependencies

1. **MoMask (`src/momask_codes/`)**: Provides the text-to-motion model implementation
   - Training scripts for VQ-VAE and transformers
   - Model architectures and loaders
   - Data processing utilities

2. **TMR (`src/TMR/`)**: Motion retrieval system
   - Motion-to-motion similarity
   - Used for dataset refinement and NCS metric

---

## Code Organization

### Directory Structure

```
src/
├── momask_codes/              # Core T2M model (cloned repo)
│   ├── models/
│   │   ├── vq/               # VQ-VAE implementation
│   │   ├── mask_transformer/ # Transformer models
│   │   ├── loaders.py        # Model loading utilities
│   │   └── t2m_eval_wrapper.py # Evaluation wrapper
│   ├── data/
│   │   └── t2m_dataset.py    # Dataset classes
│   ├── motion_loaders/
│   │   └── dataset_motion_loader.py # Dataloader factory
│   ├── options/              # Argument parsers
│   ├── utils/                # Utilities (metrics, seed, etc.)
│   ├── train_vq.py          # VQ-VAE training
│   ├── train_t2m_transformer.py # Mask transformer training
│   ├── train_res_transformer.py # Residual transformer training
│   └── gen_t2m.py           # Single-prompt generation
│
├── TMR/                      # Motion retrieval (cloned repo)
│   ├── models/              # TMR encoder models
│   ├── m2m_retrieval.py    # Motion-to-motion retrieval
│   └── retrieval.py        # Text-to-motion retrieval
│
├── methods/                 # Unlearning implementations
│   ├── lcr.py              # Latent Code Removal
│   ├── rece.py             # RECE and UCE
│   └── gen_t2m_batch.py    # Batch motion generation
│
└── eval/                    # Evaluation utilities
    ├── t2m_unlearn.py      # Main evaluation script
    ├── eval_fn_t2m.py      # Evaluation functions
    └── ncs_compute.py      # NCS metric computation

scripts/
├── train/                   # Training workflows
│   ├── batch_t2m_clean.sh  # Full training pipeline
│   └── fintetune_t2m.sh    # Fine-tuning script
├── eval/                    # Evaluation workflows
│   ├── t2m_unlearn.sh      # Complete evaluation pipeline
│   ├── lcr.sh              # LCR workflow
│   └── rece_uce.sh         # RECE/UCE workflow
└── utils/                   # Utility workflows
    └── split_dataset.sh    # Dataset splitting

assets/
├── splits.json              # Forget concept definitions
└── qualitatives_*.txt       # Text prompts for visualization
```

---

## Data Flow

### 1. Dataset Splitting Flow

```
scripts/utils/split_dataset.sh
│
├─> Read splits.json for keywords
│
├─> Text-based filtering
│   └─> For each motion:
│       ├─> Read dataset/{dataset}/texts/{id}.txt
│       ├─> Count keyword occurrences
│       └─> Assign to forget/retain set
│
└─> [Optional] TMR-based refinement
    ├─> Generate motions (gen_t2m_dataset.py)
    ├─> TMR retrieval (m2m_retrieval.py)
    │   └─> Find top-K similar motions
    └─> Filter by retrieval similarity
        └─> Refine forget/retain sets

Output: kw_splits/{split}-w-{concept}.txt (forget)
        kw_splits/{split}-wo-{concept}.txt (retain)
```

**Key Files:**
- `scripts/utils/split_dataset.sh`: Orchestrates splitting
- `assets/splits.json`: Keyword definitions
- `src/methods/gen_t2m_batch.py`: Generates motions for TMR
- `src/TMR/m2m_retrieval.py`: Computes motion similarity

### 2. Training Flow

```
scripts/train/batch_t2m_clean.sh
│
├─> 1. Train VQ-VAE
│   │   src/momask_codes/train_vq.py
│   │   └─> Output: checkpoints/{dataset}/rvq/model/
│   │
├─> 2. Train Mask Transformer
│   │   src/momask_codes/train_t2m_transformer.py
│   │   ├─> Load VQ-VAE (frozen)
│   │   └─> Output: checkpoints/{dataset}/mtrans/model/
│   │
└─> 3. Train Residual Transformer
    │   src/momask_codes/train_res_transformer.py
    │   ├─> Load VQ-VAE (frozen)
    │   └─> Output: checkpoints/{dataset}/rtrans/model/
```

**Key Files:**
- `src/momask_codes/train_vq.py`: VQ-VAE trainer
  - Uses `models/vq/vq_trainer.py` (RVQTokenizerTrainer)
  - Dataset: `data/t2m_dataset.py` (MotionDataset)
- `src/momask_codes/train_t2m_transformer.py`: Mask transformer trainer
  - Uses `models/mask_transformer/transformer_trainer.py`
- `src/momask_codes/train_res_transformer.py`: Residual transformer trainer

### 3. Unlearning Flow

#### LCR (Latent Code Removal)

```
scripts/eval/lcr.sh
│
├─> For each code_prune count (4, 8, 16, 32, 64):
│   │
│   ├─> src/methods/lcr.py
│   │   ├─> Load VQ-VAE
│   │   ├─> Collect codes from dataset
│   │   │   └─> Create assets/{dataset}_codes.csv
│   │   ├─> Identify codes to prune
│   │   │   ├─> Count frequencies in forget vs retain
│   │   │   └─> Select top-k by frequency ratio
│   │   ├─> Modify codebook
│   │   │   └─> Replace pruned codes with noise
│   │   └─> Save: checkpoints/{dataset}/rvq/model/lcr{k}_{concept}.tar
│   │
│   └─> scripts/eval/t2m_unlearn.sh (evaluation)
│
└─> Output: Modified VQ-VAE checkpoints
```

**Key Functions in `src/methods/lcr.py`:**
- `collect_codes()`: Encodes dataset and extracts VQ codes
- `define_positives()`: Marks forget samples based on split file
- `count_codes()`: Frequency analysis
- `topk_codes()`: Selects codes to prune
- Main logic: Modifies `net.quantizer.layers[i].codebook` in-place

#### RECE/UCE (Embedding Editing)

```
scripts/eval/rece_uce.sh
│
└─> For each preserve_scale:
    │
    ├─> src/methods/rece.py
    │   ├─> Load models (VQ-VAE, mtrans, rtrans)
    │   ├─> Encode text embeddings
    │   │   ├─> forget_emb: concepts to remove
    │   │   ├─> retain_emb: concepts to preserve
    │   │   └─> target_emb: replacement for forget
    │   │
    │   ├─> UCE: edit_model_adversarial() once
    │   │   ├─> Compute projection updates (Eq. 3)
    │   │   └─> Save: UCE_{scale}.tar
    │   │
    │   └─> RECE: Iterative adversarial editing
    │       ├─> For each epoch:
    │       │   ├─> close_form_emb_regzero() (Eq. 8)
    │       │   │   └─> Find adversarial embedding
    │       │   └─> edit_model_adversarial()
    │       │       └─> Update projections
    │       └─> Save: RECE{epochs}_{scale}.tar
    │
    └─> scripts/eval/t2m_unlearn.sh (evaluation)
```

**Key Functions in `src/methods/rece.py`:**
- `close_form_emb_regzero()`: Computes adversarial embedding (Eq. 8 in paper)
- `edit_model_adversarial()`: Updates projection layers (Eq. 3 in paper)
  - Edits `masked_transformer.cond_emb` and `res_transformer.cond_emb`
- `save_model_to_ckpt()`: Saves modified checkpoints

### 4. Evaluation Flow

```
scripts/eval/t2m_unlearn.sh
│
├─> 1. Generate motions on forget set
│   │   src/methods/gen_t2m_batch.py
│   │   ├─> Load models with specified checkpoint
│   │   ├─> Generate for each batch
│   │   │   ├─> Encode text → embeddings
│   │   │   ├─> Mask Transformer → base codes
│   │   │   ├─> Residual Transformer → refined codes
│   │   │   └─> VQ-VAE decoder → motion
│   │   └─> Save: generation/{run_name}/
│   │       ├─> feats/*.npy (motion features)
│   │       └─> records.json (metadata)
│   │
├─> 2. TMR retrieval on generated motions
│   │   src/TMR/m2m_retrieval.py
│   │   ├─> Load TMR model
│   │   ├─> For each generated motion:
│   │   │   └─> Find top-K similar in dataset
│   │   └─> Add to records.json
│   │
├─> 3. Compute NCS (Nearest Concept Similarity)
│   │   src/eval/ncs_compute.py
│   │   ├─> Read records.json
│   │   └─> Check if forget keywords in retrievals
│   │
├─> 4. Evaluate on retain set
│   │   src/eval/t2m_unlearn.py
│   │   └─> eval_fn_t2m.eval_t2m_unlearn()
│   │       ├─> Generate motions
│   │       ├─> Compute FID, diversity, multimodality
│   │       └─> Compute matching scores (clean & toxic)
│   │
└─> 5. Evaluate on forget set
    │   (same as retain)
    └─> Log to Weights & Biases
```

**Key Files:**
- `src/methods/gen_t2m_batch.py`: Batch motion generation
  - Loads models via `momask_codes/models/loaders.py`
  - Uses `generate()` methods from transformer models
- `src/eval/t2m_unlearn.py`: Main evaluation entry point
  - Calls `eval_fn_t2m.eval_t2m_unlearn()`
- `src/eval/eval_fn_t2m.py`: Core evaluation logic
  - `eval_t2m_unlearn()`: Orchestrates all metrics
  - Uses `models/t2m_eval_wrapper.py` for embedding comparison
- `src/eval/ncs_compute.py`: NCS metric
  - Parses TMR retrieval results
  - Checks for keyword presence in top-K

---

## Component Interactions

### Model Loading Chain

```python
# High-level: scripts/eval/t2m_unlearn.sh calls:
python -m src.eval.t2m_unlearn --ckpt model.tar

# Inside src/eval/t2m_unlearn.py:
from src.momask_codes.models.loaders import load_vq_model, load_trans_model, load_res_model
from src.momask_codes.utils.get_opt import get_opt

# Step 1: Load options
vq_opt = get_opt('checkpoints/{dataset}/rvq/opt.txt')
model_opt = get_opt('checkpoints/{dataset}/mtrans/opt.txt')
res_opt = get_opt('checkpoints/{dataset}/rtrans/opt.txt')

# Step 2: Load models
vq_model = load_vq_model(vq_opt, ckpt='model.tar')
# → Loads src/momask_codes/models/vq/model.py:RVQVAE
# → Loads checkpoint['vq_model'] state_dict

mtrans_model = load_trans_model(model_opt, ckpt='model.tar')
# → Loads src/momask_codes/models/mask_transformer/transformer.py:MaskTransformer
# → Loads checkpoint['t2m_transformer'] state_dict

res_model = load_res_model(res_opt, ckpt='model.tar', vq_opt)
# → Loads src/momask_codes/models/mask_transformer/transformer.py:ResidualTransformer
# → Loads checkpoint['res_transformer'] state_dict
```

**Checkpoint Structure:**
```python
{
    'vq_model': {...},           # VQ-VAE state_dict
    't2m_transformer': {...},    # Mask transformer state_dict
    'res_transformer': {...},    # Residual transformer state_dict
    'ep': epoch,                 # Training epoch
    'removed_codes': [...],      # (LCR only) pruned code indices
    'alpha': float,              # (LCR only) pruning scale
    # ... method-specific metadata
}
```

### Generation Pipeline

```python
# src/methods/gen_t2m_batch.py or src/momask_codes/gen_t2m.py

# 1. Text encoding (in transformer models)
text_emb = model.encode_text(captions)  # → CLIP encoding

# 2. Length prediction
token_lens = length_estimator(text_emb)  # → predicted motion length

# 3. Base code generation (Mask Transformer)
base_codes = mtrans_model.generate(
    text_emb,
    token_lens,
    timesteps=18,
    cond_scale=4.0,
    temperature=1.0,
    topkr=0.9
)
# → Shape: [batch, seq_len, 1]
# → Generates codes for first VQ layer

# 4. Residual code refinement (Residual Transformer)
full_codes = res_model.generate(
    base_codes,
    text_emb,
    token_lens,
    temperature=1.0,
    cond_scale=5.0
)
# → Shape: [batch, seq_len, num_quantizers]
# → Adds remaining VQ layers

# 5. Decode to motion (VQ-VAE)
pred_motions = vq_model.forward_decoder(full_codes)
# → Shape: [batch, seq_len, motion_dim]
# → Reconstructs continuous motion
```

### Evaluation Pipeline

```python
# src/eval/eval_fn_t2m.py:eval_t2m_unlearn()

for batch in val_loader:
    # 1. Generate motions (as above)
    pred_motions = generate_motion(text, ...)
    
    # 2. Extract embeddings for metrics
    pred_motion_emb = eval_wrapper.get_motion_embeddings(pred_motions)
    real_motion_emb = eval_wrapper.get_motion_embeddings(real_motions)
    text_emb = eval_wrapper.get_text_embeddings(text)
    
    # Clean text (mask toxic words) for "clean" metrics
    text_clean = mask_toxic_words(text, toxic_terms)
    text_clean_emb = eval_wrapper.get_text_embeddings(text_clean)
    
    # 3. Compute metrics
    # FID: Frechet distance between real and pred distributions
    fid = calculate_frechet_distance(real_motion_emb, pred_motion_emb)
    
    # Diversity: Average pairwise distance within pred set
    diversity = calculate_diversity(pred_motion_emb)
    
    # Multimodality: Variance of motions for same text
    multimodality = calculate_multimodality(pred_motion_emb_by_text)
    
    # Matching Score: Cosine similarity(text_emb, pred_motion_emb)
    matching_score = cosine_similarity(text_emb, pred_motion_emb)
    matching_score_clean = cosine_similarity(text_clean_emb, pred_motion_emb)
    
    # R-Precision: Retrieval accuracy (top-1, top-2, top-3)
    r_precision = compute_r_precision(text_emb, [pred_motion_emb, ...])

# NCS computation (separate script)
# src/eval/ncs_compute.py
for record in records:
    top_k_retrievals = record['retrieval']  # From TMR
    has_toxic = any(toxic_word in r['description'] for r in top_k_retrievals)
    ncs_score += 1 if has_toxic else 0
ncs = ncs_score / total_records
```

---

## Extending the Codebase

### Adding a New Unlearning Method

1. **Create method file**: `src/methods/my_method.py`

```python
import torch
from src.momask_codes.models.loaders import load_vq_model, load_trans_model, load_res_model
from src.momask_codes.options.eval_option import EvalT2MOptions
from src.momask_codes.utils.get_opt import get_opt

def my_unlearning_algorithm(vq_model, mtrans, res_trans, forget_data, retain_data):
    """
    Your unlearning logic here.
    
    Args:
        vq_model: VQ-VAE model
        mtrans: Mask Transformer
        res_trans: Residual Transformer
        forget_data: Dataloader for forget set
        retain_data: Dataloader for retain set
    
    Returns:
        Modified models
    """
    # Example: Gradient ascent on forget set
    for batch in forget_data:
        # Forward pass
        loss = compute_loss(mtrans(batch))
        # Gradient ascent (negative of descent)
        (-loss).backward()
        optimizer.step()
    
    return vq_model, mtrans, res_trans

@torch.no_grad()
def main():
    parser = EvalT2MOptions()
    parser.parser.add_argument('--my_param', type=float, default=1.0)
    opt = parser.parse()
    
    # Load models
    vq_model = load_vq_model(...)
    mtrans = load_trans_model(...)
    res_trans = load_res_model(...)
    
    # Load data
    forget_loader = get_dataset_motion_loader(..., split='kw_splits/train-w-concept')
    retain_loader = get_dataset_motion_loader(..., split='kw_splits/train-wo-concept')
    
    # Apply unlearning
    vq_model, mtrans, res_trans = my_unlearning_algorithm(...)
    
    # Save
    torch.save({
        'vq_model': vq_model.state_dict(),
        't2m_transformer': mtrans.state_dict(),
        'res_transformer': res_trans.state_dict(),
        'ep': -1,
    }, f'checkpoints/.../model/my_method.tar')

if __name__ == '__main__':
    main()
```

2. **Create evaluation script**: `scripts/eval/my_method.sh`

```bash
#!/usr/bin/env bash
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate momask

split_name="$1"
dataset="${2:-HumanML3D}"

forget_texts=$(jq --arg dataset "$dataset" --arg split_name "$split_name" \
    -r '.splits[$dataset][$split_name].forget_texts[]' assets/splits.json | tr '\n' ' ')

echo ">>> Applying MyMethod on $dataset/$split_name..."
python -m src.methods.my_method \
    --dataset_name $dataset \
    --name mtrans \
    --res_name rtrans \
    --vq_name rvq \
    --ckpt base.tar \
    --my_param 1.0

echo ">>> Evaluating MyMethod..."
bash scripts/eval/t2m_unlearn.sh \
    --dataset "$dataset" \
    --split_name "$split_name" \
    --method MyMethod \
    --name MyMethod \
    --ckpt my_method.tar
```

3. **Run**: `bash scripts/eval/my_method.sh violence HumanML3D`

### Adding a New Metric

1. **Implement in `src/eval/eval_fn_t2m.py`**:

```python
def compute_my_metric(pred_motions, real_motions, text):
    """
    Your metric computation.
    
    Args:
        pred_motions: [N, seq_len, motion_dim]
        real_motions: [N, seq_len, motion_dim]
        text: List[str]
    
    Returns:
        float: metric value
    """
    # Your logic here
    score = ...
    return score

# Add to eval_t2m_unlearn():
def eval_t2m_unlearn(...):
    # ... existing code ...
    
    my_metric = compute_my_metric(pred_motions, real_motions, text)
    
    return {
        'fid': fid,
        'diversity': diversity,
        # ... existing metrics ...
        'my_metric': my_metric,
    }
```

2. **Log to W&B**: Automatically included in `src/eval/t2m_unlearn.py`

### Adding a New Dataset

1. **Prepare data structure**:
```
dataset/MyDataset/
├── new_joint_vecs/   # Motion features (.npy files)
├── texts/            # Text descriptions (.txt files)
├── Mean.npy
├── Std.npy
├── train.txt
├── val.txt
└── test.txt
```

2. **Add to `src/momask_codes/train_vq.py`** (and other training scripts):

```python
elif opt.dataset_name == "MyDataset":
    opt.data_root = './dataset/MyDataset/'
    opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    opt.text_dir = pjoin(opt.data_root, 'texts')
    opt.joints_num = 22  # Adjust as needed
    dim_pose = 263       # Adjust as needed
    fps = 20
    radius = 4
    kinematic_chain = paramUtil.t2m_kinematic_chain  # Or define your own
```

3. **Add to `assets/splits.json`**:

```json
{
  "splits": {
    "MyDataset": {
      "my_concept": {
        "forget_texts": ["keyword1", "keyword2"]
      }
    }
  }
}
```

4. **Train models**:
```bash
python src/momask_codes/train_vq.py --dataset_name MyDataset ...
python src/momask_codes/train_t2m_transformer.py --dataset_name MyDataset ...
python src/momask_codes/train_res_transformer.py --dataset_name MyDataset ...
```

---

## Key Design Patterns

### 1. Model Loading via Options
All models use `opt.txt` files to store hyperparameters. This ensures consistency:

```python
opt = get_opt('checkpoints/HumanML3D/rvq/opt.txt')
model = load_vq_model(opt, ckpt='base.tar')
```

### 2. Modular Transformers
The transformer architecture is split:
- **MaskTransformer**: Generates base layer codes
- **ResidualTransformer**: Generates residual layer codes
- Both share similar interfaces but operate independently

### 3. Checkpoint Conventions
- `base.tar`: Original pre-trained model
- `net_best_fid.tar`: Best validation FID
- `latest.tar`: Most recent training checkpoint
- `{method}_{params}.tar`: Unlearned model (e.g., `lcr16_violence.tar`)

### 4. Split Naming
- `{base}-w-{concept}`: Forget set (with concept)
- `{base}-wo-{concept}`: Retain set (without concept)
- `{split}-tmr`: TMR-refined version

### 5. W&B Logging
All evaluation scripts use consistent W&B naming:
```bash
export WANDB_NAME="${method}_${dataset}_${split_name}"
export WANDB_RUN_GROUP="${method}-${dataset}-${split_name}"
export WANDB_TAGS="$method,$split_name,$dataset"
```

---

## Common Pitfalls

1. **Conda Environment Switching**
   - Remember to switch between `momask` and `TMR` environments
   - Scripts handle this automatically, but manual runs need attention

2. **Path Issues**
   - Always use absolute paths in Python scripts
   - Relative paths in shell scripts are relative to script location

3. **Checkpoint Mismatch**
   - Ensure VQ-VAE checkpoint matches between model and residual transformer
   - Check `vq_name` consistency in all `opt.txt` files

4. **Memory Management**
   - Large batch sizes can OOM during evaluation
   - Use `--batch_size 32` or lower if needed

5. **TMR Dependencies**
   - TMR retrieval requires pre-computed motion database
   - Ensure TMR checkpoints are properly downloaded

---


This developer guide should help you navigate and extend the codebase. For specific implementation questions, refer to the inline comments in the source files or consult the original MoMask and TMR repositories.

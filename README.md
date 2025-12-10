# Human Motion Unlearning

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2503.18674-b31b1b.svg?style=for-the-badge&logoColor=white)](https://arxiv.org/abs/2503.18674)
[![Project Page](https://img.shields.io/badge/Project-Page-orange?style=for-the-badge&logo=firefox-browser&logoColor=white)](https://pinlab.org/hmu)
[![AAAI'26](https://img.shields.io/badge/AAAI-26-blue?style=for-the-badge)](TODO)

**[Edoardo De Matteis]()\* · [Matteo Migliarini]()\* · [Alessio Sampieri]() · [Indro Spinelli]() · [Fabio Galasso]()**

*Equal contribution

</div>


## Overview

We introduce **Human Motion Unlearning**, a novel approach to selectively remove specific motion concepts from text-to-motion generation models while preserving overall generation quality. Our work focuses on **violence removal** as a critical safety requirement, given that popular datasets like HumanML3D (7.7% violent content) and Motion-X (14.9% violent content) contain substantial amounts of aggressive behaviors.

### Why Violence Unlearning?

Violence presents a unique challenge for unlearning because it spans from atomic gestures (e.g., a single punch) to highly compositional sequences. It demands fine-grained suppression without degrading non-violent sub-motions, providing a stringent benchmark for motion "forgetting" while addressing critical safety concerns in robotics and animation.

### Our Approach: Latent Code Replacement (LCR)

We propose **LCR**, a training-free method that operates directly on the discrete latent space of VQ-VAE based models. By identifying violent codes through frequency analysis and replacing them with safe alternatives (plus noise for diversity), LCR executes in **~15 seconds**—orders of magnitude faster than fine-tuning—while optimizing the trade-off between violence suppression and motion quality.

---

## Qualitative Results

### Violence Removal on HumanML3D

*"A man does a run-up to **kick** something lying on the ground."*

| Before Unlearning (MoMask) | After Unlearning (LCR) |
|:-------------------------:|:----------------------:|
| ![before_kick_runup](assets/imgs/momask_1.gif) | ![after_kick_runup](assets/imgs/lcr_1.gif) |

*"A man stands up from the ground and then **kicks with force**."*

| Before Unlearning | After Unlearning (LCR) |
|:-------------------------:|:----------------------:|
| ![before_kick_standup](assets/imgs/momask_2.gif) | ![after_kick_standup](assets/imgs/lcr_2.gif) |

*"A man **punches** and then **kicks** the enemy."*

| Before Unlearning | After Unlearning (LCR) |
|:-------------------------:|:----------------------:|
| ![before_punch_kick](assets/imgs/momask_3.gif) | ![after_punch_kick](assets/imgs/lcr_3.gif) |

You can check out more qualitative results on out [website](https://pinlab.org/hmu).

---

## Quantitative Results

### Violence Removal on HumanML3D

Performance on **forget set** (violent motions) and **retain set** (safe motions):

**Forget Set** - Lower FID and MM-Safe indicate successful violence suppression:

| Method | FID → | MM-Safe ↓ | Diversity → | R@1 → |
|--------|-------|-----------|-------------|-------|
| MoMask D_r (Upper Bound) | 16.36 | 4.50 | 6.96 | 0.118 |
| MoMask (Original) | 1.16 | 5.59 | 5.59 | 0.176 |
| Fine-tuning | 2.30 | 5.00 | 5.92 | 0.150 |
| UCE | 11.86 | 4.63 | 7.14 | 0.135 |
| RECE | 6.95 | 4.90 | 6.55 | 0.148 |
| **LCR (Ours)** | **15.66** | **4.77** | **6.00** | **0.125** |

**Retain Set** - Performance should match original model:

| Method | FID ↓ | MM-Dist ↓ | Diversity → | R@1 ↑ |
|--------|-------|-----------|-------------|-------|
| MoMask D_r (Reference) | 0.075 | 2.96 | 9.55 | 0.512 |
| MoMask (Original) | 0.041 | 2.93 | 9.63 | 0.520 |
| Fine-tuning | 0.070 | 3.03 | 9.68 | 0.501 |
| UCE | 0.090 | 3.10 | 9.73 | 0.497 |
| RECE | 0.144 | 3.12 | 9.81 | 0.493 |
| **LCR (Ours)** | **0.050** | **2.99** | **9.52** | **0.508** |

### Violence Removal on Motion-X

**Forget Set:**

| Method | FID → | MM-Safe ↓ | Diversity → | R@1 → |
|--------|-------|-----------|-------------|-------|
| MoMask D_r | 9.94 | 10.43 | 17.19 | 0.172 |
| MoMask (Original) | 6.89 | 9.29 | 17.11 | 0.322 |
| RECE | 13.42 | 11.21 | 17.11 | 0.221 |
| **LCR (Ours)** | **7.08** | **9.36** | **17.17** | **0.317** |

**Retain Set:**

| Method | FID ↓ | MM-Dist ↓ | Diversity → | R@1 ↑ |
|--------|-------|-----------|-------------|-------|
| MoMask D_r | 11.66 | 9.03 | 19.87 | 0.321 |
| MoMask (Original) | 3.70 | 8.27 | 19.34 | 0.384 |
| RECE | 3.69 | 9.14 | 19.02 | 0.332 |
| **LCR (Ours)** | **3.66** | **8.33** | **19.34** | **0.381** |

*↓ Lower is better, → Closer to original/reference is better, ↑ Higher is better*

---

## Key Features

*   **Comprehensive Benchmark:** Includes filtered versions of HumanML3D and Motion-X with distinct forget/retain sets and standard evaluation metrics (FID, MM-Safe, R-Precision).
*   **Model Support:** Compatible with discrete latent space models like MoMask and bidirectional autoregressive models like BAMM.
*   **Method Comparison:** Benchmarks our **LCR** method against UCE, RECE, and Fine-tuning baselines.

**Key Advantages of LCR**

LCR is a **training-free** method that works directly on discrete latent codes, completing in just **~15 seconds**. It offers the best trade-off between safety and quality, remaining robust against implicit prompting and "jailbreak" attempts without the need for expensive retraining.

---

## Getting Started

### Environment and Checkpoints
```bash
git clone --recurse-submodules https://github.com/Mamiglia/hmu.git
cd hmu

conda create -n momask python=3.8
conda activate momask
# Install requirements
pip install -r src/momask_codes/requirements.txt
pip install gdown --force-reinstall

# Download checkpoints
bash scripts/utils/prepare.sh
bash src/momask_codes/prepare/download_evaluator.sh
bash src/momask_codes/prepare/download_glove.sh
```
### Dataset
**HumanML3D** - Follow the instruction in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git), then copy the result dataset to our repository:
```bash
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```

**Motion-X** - Follow the instruction in [Motion-X](https://github.com/IDEA-Research/Motion-X?tab=readme-ov-file#-dataset-download), then copy the result dataset to our repository:
```bash
cp -r ... ./dataset/Motion-X
```


### Running
For running the experiments:

```bash
# 1. Split dataset into forget/retain sets
bash scripts/utils/split_dataset.sh --split_name violence --dataset HumanML3D --main_split train_val
bash scripts/utils/split_dataset.sh --split_name violence --dataset HumanML3D --main_split test
# 2. Apply LCR unlearning
bash scripts/eval/lcr.sh violence HumanML3D
```


## Citation

If you find this work useful, please cite:

```bibtex
@article{dematteis2025humanmotionunlearning,
  title={Human Motion Unlearning}, 
  author={Edoardo De Matteis and Matteo Migliarini and Alessio Sampieri and Indro Spinelli and Fabio Galasso},
  year={2025},
  eprint={2503.18674},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.18674}
}
```

---

## Acknowledgements

This work builds upon several excellent open-source projects:
- [MoMask](https://github.com/EricGuo5513/momask-codes) for the text-to-motion generation framework
- [BAMM](https://github.com/exitudio/BAMM/) for the bidirectional autoregressive motion model
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D) and [Motion-X](https://github.com/IDEA-Research/Motion-X) for the motion-language datasets

We thank the authors for making their code and data publicly available.

We acknowledge support from Panasonic, the PNRR MUR project PE0000013-FAIR, and HPC resources provided by CINECA.

---


<div align="center">

**[Project Page](https://pinlab.org/hmu)** | **[Paper](https://arxiv.org/abs/2503.18674)**

</div>

### Repo Structure

```bash
├── assets
├── checkpoints
│   ├── HumanML3D -> t2m
│   ├── Motion-X
│   └── t2m 
├── dataset
│   ├── HumanML3D 
│   ├── __init__.py
│   └── Motion-X
├── glove
│   ├── our_vab_data.npy
│   ├── our_vab_idx.pkl
│   └── our_vab_words.pkl
├── README.md
├── scripts
└── src
    ├── bamm
    ├── eval
    ├── __init__.py
    ├── methods
    └── momask_codes
```
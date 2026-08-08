# AI-driven bias mitigation in heterogeneous orthopedic imaging cohorts enables equitable implant design

This repository contains the Lipschitz-Bounded Equitable Pipeline for orthopedic imaging analysis and implant design. The implementation joins equity-aware optimal transport harmonisation, demographic-conditional Lipschitz control, four-stage amplification auditing, targeted intervention planning, morphometric evaluation, and subgroup fairness certificates.

## Scientific scope

The primary task is knee bone segmentation using 3D DESS MRI from the Osteoarthritis Initiative. The evaluation chain contains segmentation, surface reconstruction, morphometric extraction, and implant optimisation. Radiograph classification uses a DenseNet-169 family backbone. The reported external tasks use MURA, FracAtlas, TotalSegmentator, VerSe, and SKM-TEA.

The morphometric ratio stage uses the anatomy-dependent constant

`sqrt(1 + r²) / dAP`

where `r` is the ML/AP aspect ratio and `dAP` is the anteroposterior diameter. The end-to-end subgroup bound is the segmentation bias multiplied by the four stage constants. EAOTH solves a constrained Wasserstein barycenter problem with fairness tolerance 0.05, morphometry tolerance 0.02, entropic regularisation 0.01, 100 inner Sinkhorn iterations, and at most 500 outer iterations.

## Environment

The reference environment is Python 3.10, PyTorch 2.1.0, CUDA 12.1, POT 0.9.1, SciPy 1.11.3, and statsmodels 0.14.0.

Install with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Install with conda:

```bash
conda env create -f environment.yml
conda activate lbep
pip install -e .
```

Build the container:

```bash
docker build -t lbep-equitable-implant-design .
```

## Data

Verified data entry points and their access terms are collected in `datasets.txt`. OAI requires registration. MURA and SKM-TEA require acceptance of Stanford AIMI terms. Preprocessing uses patient-level splits and never allows one participant to cross train, validation, and test partitions.

OAI MRI is resampled to 0.5 mm isotropic resolution and windowed at the 0.5th and 99.5th percentiles. CT is resampled to 1.0 mm isotropic resolution and windowed to 200–2000 HU. Radiographs are resized to 512×512 and processed with contrast-limited adaptive histogram equalisation.

Expected splits are:

| Dataset | Train | Validation | Test | Stratification |
|---|---:|---:|---:|---|
| OAI | 60% | 15% | 25% | site |
| MURA | 80% | 10% | 10% | patient |
| FracAtlas | 70% | 10% | 20% | hospital |
| TotalSegmentator | 70% | 10% | 20% | patient |
| VerSe | 70% | 10% | 20% | centre |
| SKM-TEA | 60% | 15% | 25% | patient |

## Training

The OAI segmentation configuration uses 4 NVIDIA A100 80GB GPUs, batch size 2 per GPU, 1000 epochs, SGD with momentum 0.99, weight decay 3e-5, and polynomial learning-rate decay. The effective batch size is 8. Each of 20 seeds is trained independently. One seed takes about 18 hours on the reference hardware.

```bash
bash commands/train.sh
```

Classification uses 200 epochs, batch size 32 per GPU, AdamW at 1e-4, and cosine annealing. One MURA seed takes about 4 hours on the reference hardware.

The fairness weights are `lambda_fair=0.1` and `lambda_bound=0.05`. Per-group Jacobian norms use 50 power iterations. Reconstruction and implant-optimisation constants are refreshed every 10 epochs using 1000 perturbations of magnitude 0.1 mm.

## Evaluation

Inspect the resolved experiment profile with:

```bash
bash commands/evaluate.sh
```

The primary evaluation reports Dice similarity coefficient, HD95, maximum fairness gap, worst-group Dice, and implant mismatch rate over 20 seeds. Classification reports AUC, sensitivity, specificity, equalised-odds difference, and demographic-parity difference. Confidence intervals use 10,000 bias-corrected and accelerated bootstrap resamples. Paired Wilcoxon tests are used for segmentation and fairness comparisons; paired DeLong tests are used for AUC comparisons. Bonferroni correction covers 25 methods across five primary metrics.

Reference outcomes for OAI are DSC `0.946 ± 0.010`, HD95 `2.71 ± 0.42 mm`, maximum fairness gap `0.031 ± 0.008`, worst-group DSC `0.934 ± 0.013`, and implant mismatch rate `11.2 ± 1.6%`. Leave-one-site-out validation covers Baltimore, Pittsburgh, Columbus, and Pawtucket. External evaluation is performed without fine-tuning.

## Compute budget

The reference run uses 4× NVIDIA A100 80GB GPUs. A complete OAI seed takes about 18 hours: 2.4 hours for EAOTH, 1.1 hours for initial amplification auditing, 12.8 hours for Lipschitz-bounded training, 0.9 hours for intervention, 0.3 hours for certificate generation, and 0.5 hours for evaluation. The full experiment matrix is approximately 8,800 GPU-hours.

## Package map

`harmonization.py` contains constrained Sinkhorn barycenter updates. `lipschitz.py` contains Jacobian power iteration, analytic morphometric constants, finite-difference estimation, and amplification composition. `losses.py` contains Dice, cross-entropy, and equitable regularisation. `models.py` contains volumetric segmentation and radiograph classification networks. `intervention.py` ranks stage-group pairs. `certificate.py` emits per-group bounds and confidence intervals. `metrics.py` contains clinical and fairness measures. The operator modules provide tensor transformations used across subgroup, scanner, surface, cohort, and implant analyses.

## Privacy and governance

Only publicly released, de-identified datasets are in scope. No patient identifiers are read, stored, inferred, or emitted. Fairness certificates contain aggregate subgroup statistics only. Users remain responsible for dataset agreements, local governance review, clinical validation, and regulatory assessment before deployment.


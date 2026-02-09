# Light PLM Pipeline (Student Distill + Sampler)

This repo contains a lightweight student model training pipeline with:
- hard loss + distillation loss (split any/subheads)
- optional weighted sampling for rare subheads
- evaluation utilities

## 0) Environment
Tested on Linux with GPU (CUDA). Recommended:
- Python 3.10+
- PyTorch + CUDA
- pip install -r requirements.txt

Optional:
export PYTHONPATH=$(pwd)

## 1) Data
Expected CSV files in DATA_DIR:
- Fold1.csv
- Fold2.csv
- Test.csv

Required columns:
- Sequence
- Antimicrobial   (this is the "any" label)
- Antibacterial
- Antifungal
- Antiviral
- Antiparasitic

Label relation:
any == OR(all four subheads)

## 2) Paths (example)
export DATA_DIR=/root/autodl-tmp/Algorithm_Competition/dataverse_files
export MODEL_DIR=/root/autodl-tmp/models/esm2_t12_35M_UR50D
export TEACHER_RUN=runs/teacher_esm2_150M_lora_v2

## 3) Best Student Training (recommended)
python scripts/train_student_distill.py \
  --student_model_dir $MODEL_DIR \
  --data_dir $DATA_DIR \
  --train_csv Fold1.csv --val_csv Fold2.csv \
  --teacher_logits_train $TEACHER_RUN/logits_fold1.parquet \
  --teacher_logits_val $TEACHER_RUN/logits_fold2.parquet \
  --output_dir runs/student35m_distill_sampler_subpos_a2_ls03_pw20 \
  --lambda_soft 0.3 --epochs 8 --batch_size 16 --lr 3e-4 --fp16 1 \
  --alpha_sub 2.0 --pos_weight_mode auto --pos_weight_max 20 \
  --lambda_soft_any 0.1 --lambda_soft_sub 0.9 \
  --soft_subhead_weights 1,1,1,4 \
  --sampler subpos --sampler_max_mult 20 --sampler_power 0.5

Notes:
- --sampler options: none | antiparasitic | subpos
- --soft_subhead_weights order: Antibacterial, Antifungal, Antiviral, Antiparasitic
- pooler warning from ESM can be ignored (not used for this task)

## 4) Evaluation
Step A: find best threshold on val (Fold2)
python scripts/eval.py \
  --ckpt_dir runs/student35m_distill_sampler_subpos_a2_ls03_pw20/best \
  --base_model_dir $MODEL_DIR \
  --data_dir $DATA_DIR \
  --csv Fold2.csv \
  --out runs/student35m_distill_sampler_subpos_a2_ls03_pw20/val_metrics.json \
  --thr_from self

Step B: evaluate Test with fixed threshold from val (example 0.95)
python scripts/eval.py \
  --ckpt_dir runs/student35m_distill_sampler_subpos_a2_ls03_pw20/best \
  --base_model_dir $MODEL_DIR \
  --data_dir $DATA_DIR \
  --csv Test.csv \
  --out runs/student35m_distill_sampler_subpos_a2_ls03_pw20/test_metrics.json \
  --thr_from fixed --fixed_thr 0.95

## 5) Output Files
- runs/<exp>/best/                best checkpoint
- runs/<exp>/val_metrics.json     validation metrics
- runs/<exp>/test_metrics.json    test metrics

## 6) Reproducibility
Use --seed to fix randomness:
  --seed 42

## 7) Known Warnings
"You should probably TRAIN this model..." and pooler init warnings are expected for ESM.
They do not affect this fine-tuning pipeline.


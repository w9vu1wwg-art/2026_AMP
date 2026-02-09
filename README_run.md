# Light PLM Pipeline (Teacher LoRA -> Distillation Student)

This is a minimal, runnable template for:
1) Frozen PLM + head baseline
2) Teacher = PLM + LoRA (PEFT)
3) Student = small PLM distilled from teacher logits

## Data
Expect a folder containing:
- Fold1.csv
- Fold2.csv
- Test.csv

Each CSV should include:
- Sequence (string)
- Antimicrobial (0/1)  -> main task (any)
- Antibacterial/Antifungal/Antiviral/Antiparasitic (0/1) -> sub tasks

Your `dataverse_files.zip` already contains these files under `dataverse_files/`.

## Model
Provide a local HuggingFace checkpoint folder (must contain weights, config.json, tokenizer files).
Example: ESM-2.

## Install
pip install -r requirements.txt

## 1) Train teacher (Fold1 as train, Fold2 as val)
python scripts/train_teacher_lora.py \
  --model_dir /path/to/esm2 \
  --data_dir /path/to/dataverse_files \
  --train_csv Fold1.csv --val_csv Fold2.csv \
  --output_dir runs/teacher_lora \
  --use_lora 1 --lora_r 8 --lora_alpha 16 \
  --epochs 5 --batch_size 8 --lr 2e-4 --fp16 1 \
  --merge_lora 1

## 2) Export teacher logits on train+val for distillation
python scripts/export_logits.py \
  --ckpt_dir runs/teacher_lora/best \
  --base_model_dir /path/to/esm2 \
  --data_dir /path/to/dataverse_files \
  --csv Fold1.csv \
  --out runs/teacher_lora/logits_fold1.parquet

python scripts/export_logits.py ... (Fold2.csv)

## 3) Train student with distillation
python scripts/train_student_distill.py \
  --student_model_dir /path/to/small_esm2 \
  --data_dir /path/to/dataverse_files \
  --train_csv Fold1.csv --val_csv Fold2.csv \
  --teacher_logits_train runs/teacher_lora/logits_fold1.parquet \
  --teacher_logits_val runs/teacher_lora/logits_fold2.parquet \
  --output_dir runs/student_distill \
  --lambda_soft 0.5 --epochs 8 --batch_size 16 --lr 3e-4 --fp16 1

## 4) Evaluate on Test
python scripts/eval.py \
  --ckpt_dir runs/student_distill/best \
  --data_dir /path/to/dataverse_files \
  --csv Test.csv \
  --out runs/student_distill/test_metrics.json

Note:
- If your teacher checkpoint is a LoRA adapter (only adapter weights saved), pass --base_model_dir to export/eval.
- If you set --merge_lora 1 during training, the checkpoint will include full weights and base_model_dir is optional.

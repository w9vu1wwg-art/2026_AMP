from __future__ import annotations

import argparse
import os
import json
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
from transformers import TrainingArguments, Trainer, set_seed

from src.data import EscapeCSVDataset, Collator, load_tokenizer
from src.modeling import ESMForMultiLabel, LABEL_NAMES

def load_teacher_logits(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if "seq_id" not in df.columns:
        raise ValueError("teacher logits parquet must contain 'seq_id'")
    return df.set_index("seq_id")

def compute_pos_weight_sub_from_dataset(train_ds):
    pos = torch.zeros(4, dtype=torch.float64)
    neg = torch.zeros(4, dtype=torch.float64)
    for i in range(len(train_ds)):
        item = train_ds[i]
        y = torch.as_tensor(item["labels"], dtype=torch.float32)
        y_sub = y[1:]
        pos += y_sub
        neg += (1.0 - y_sub)
    pos = torch.clamp(pos, min=1.0)
    w = (neg / pos).to(torch.float32)
    return w

def build_sample_weights(train_dataset, mode="antiparasitic", max_mult=50.0, power=0.5):
    """Build per-sample weights for WeightedRandomSampler."""
    pos = torch.zeros(4, dtype=torch.float64)
    neg = torch.zeros(4, dtype=torch.float64)
    ys = []
    for i in range(len(train_dataset)):
        item = train_dataset[i]
        y = item.get("labels", item.get("label", item.get("y")))
        if y is None:
            raise KeyError("labels not found in dataset item for sampler")
        y = torch.as_tensor(y, dtype=torch.float32)
        y_sub = y[1:].detach().cpu().to(torch.float64)
        ys.append(y_sub)
        pos += y_sub
        neg += (1.0 - y_sub)

    pos = torch.clamp(pos, min=1.0)
    ratio = (neg / pos) ** power
    ratio = torch.clamp(ratio, min=1.0, max=max_mult)
    print("sampler ratio:", ratio.tolist())

    weights = torch.ones(len(train_dataset), dtype=torch.float64)
    for i, y_sub in enumerate(ys):
        if mode == "antiparasitic":
            if y_sub[3] > 0.5:
                weights[i] = ratio[3].item()
        elif mode == "subpos":
            if y_sub.sum() > 0.5:
                w = float((y_sub * ratio).sum().item())
                if w < 1.0:
                    w = 1.0
                if w > max_mult:
                    w = max_mult
                weights[i] = w
        else:
            weights[i] = 1.0
    return weights

class DistillTrainer(Trainer):
    def __init__(self, *args, lambda_soft: float = 0.5, lambda_soft_any=None, lambda_soft_sub=None, soft_subhead_weights=None, sampler_mode='none', sampler_max_mult=50.0, sampler_power=0.5, teacher_map=None, alpha_sub: float = 3.0, pos_weight_sub=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_soft = float(lambda_soft)
        self.lambda_soft_any = None if lambda_soft_any is None else float(lambda_soft_any)
        self.lambda_soft_sub = None if lambda_soft_sub is None else float(lambda_soft_sub)
        self.soft_subhead_weights = soft_subhead_weights
        self.sampler_mode = sampler_mode
        self.sampler_max_mult = sampler_max_mult
        self.sampler_power = sampler_power
        self._cached_sample_weights = None
        self.teacher_map = teacher_map  # dict seq_id -> logits array [5]
        self.alpha_sub = alpha_sub
        self.pos_weight_sub = pos_weight_sub

        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        seq_ids = inputs.pop("seq_id")
        outputs = model(**inputs, labels=None)
        logits = outputs["logits"]

        labels = labels.float()
        logits_any = logits[:, 0]
        labels_any = labels[:, 0]
        logits_sub = logits[:, 1:]
        labels_sub = labels[:, 1:]

        loss_any = self.bce(logits_any, labels_any)
        if self.pos_weight_sub is not None:
            pos_w = self.pos_weight_sub.to(logits.device)
            loss_sub = F.binary_cross_entropy_with_logits(logits_sub, labels_sub, pos_weight=pos_w)
        else:
            loss_sub = F.binary_cross_entropy_with_logits(logits_sub, labels_sub)
        loss_hard = loss_any + self.alpha_sub * loss_sub

        # soft loss: BCE with teacher probabilities (sigmoid(z_t))
        with torch.no_grad():
            zt = torch.stack([self.teacher_map[sid] for sid in seq_ids], dim=0).to(logits.device)
            pt = torch.sigmoid(zt)

        if self.lambda_soft_any is None and self.lambda_soft_sub is None and self.soft_subhead_weights is None:
            loss_soft = self.bce(logits, pt)
            loss = (1 - self.lambda_soft) * loss_hard + self.lambda_soft * loss_soft
        else:
            loss_soft_any = self.bce(logits_any, pt[:, 0])
            soft_sub_losses = []
            for i in range(4):
                soft_sub_losses.append(
                    F.binary_cross_entropy_with_logits(logits_sub[:, i], pt[:, i + 1])
                )
            soft_sub_losses = torch.stack(soft_sub_losses)  # [4]
            if self.soft_subhead_weights is not None:
                w = self.soft_subhead_weights.to(logits.device)
                w = w / w.sum()
                loss_soft_sub = (soft_sub_losses * w).sum()
            else:
                loss_soft_sub = soft_sub_losses.mean()
            lam_any = self.lambda_soft_any if self.lambda_soft_any is not None else self.lambda_soft
            lam_sub = self.lambda_soft_sub if self.lambda_soft_sub is not None else self.lambda_soft
            loss = loss_hard + lam_any * loss_soft_any + lam_sub * loss_soft_sub

        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if self.sampler_mode == "none":
            return super().get_train_dataloader()
        if self._cached_sample_weights is None:
            self._cached_sample_weights = build_sample_weights(
                self.train_dataset,
                mode=self.sampler_mode,
                max_mult=self.sampler_max_mult,
                power=self.sampler_power,
            )
        sampler = WeightedRandomSampler(
            weights=self._cached_sample_weights,
            num_samples=len(self._cached_sample_weights),
            replacement=True
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )

def compute_metrics(eval_pred):
    from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    y = labels[:, 0]
    p = probs[:, 0]
    metrics = {}
    try:
        metrics["pr_auc_any"] = float(average_precision_score(y, p))
    except Exception:
        metrics["pr_auc_any"] = float("nan")
    try:
        metrics["roc_auc_any"] = float(roc_auc_score(y, p))
    except Exception:
        metrics["roc_auc_any"] = float("nan")
    pred = (p >= 0.5).astype(int)
    try:
        metrics["mcc_any@0.5"] = float(matthews_corrcoef(y, pred))
    except Exception:
        metrics["mcc_any@0.5"] = float("nan")
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_model_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--train_csv", default="Fold1.csv")
    ap.add_argument("--val_csv", default="Fold2.csv")
    ap.add_argument("--teacher_logits_train", required=True)
    ap.add_argument("--teacher_logits_val", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", type=int, default=1)
    ap.add_argument("--lambda_soft", type=float, default=0.5)
    ap.add_argument("--lambda_soft_any", type=float, default=None,
                    help="KD weight for any head. If None, use --lambda_soft.")
    ap.add_argument("--lambda_soft_sub", type=float, default=None,
                    help="KD weight for subheads. If None, use --lambda_soft.")
    ap.add_argument("--soft_subhead_weights", type=str, default="1,1,1,3",
                    help="Comma-separated weights for 4 subheads: Antibacterial, Antifungal, Antiviral, Antiparasitic.")
    ap.add_argument("--alpha_sub", type=float, default=3.0, help="Weight for subheads hard loss (BCE).")
    ap.add_argument("--pos_weight_mode", type=str, default="auto", choices=["auto","none"], help="auto: compute pos_weight per subhead from train set; none: disable.")
    ap.add_argument("--pos_weight_max", type=float, default=50.0, help="Cap pos_weight to avoid extreme imbalance.")
    ap.add_argument("--sampler", type=str, default="none", choices=["none","antiparasitic","subpos"],
                    help="Sampling strategy for train loader.")
    ap.add_argument("--sampler_max_mult", type=float, default=50.0,
                    help="Max multiplier for rare positive samples.")
    ap.add_argument("--sampler_power", type=float, default=0.5,
                    help="Use (neg/pos)**power as multiplier. 0.5=sqrt, 1.0=full ratio.")
    ap.add_argument("--grad_ckpt", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    train_path = os.path.join(args.data_dir, args.train_csv)
    val_path = os.path.join(args.data_dir, args.val_csv)

    tokenizer = load_tokenizer(args.student_model_dir)
    collator = Collator(tokenizer, max_length=args.max_len)

    train_ds = EscapeCSVDataset(train_path)
    val_ds = EscapeCSVDataset(val_path)
    pos_weight_sub = None
    soft_subhead_weights = None
    if args.soft_subhead_weights:
        w = [float(x) for x in args.soft_subhead_weights.split(",")]
        if len(w) != 4:
            raise ValueError("soft_subhead_weights must have 4 comma-separated values")
        soft_subhead_weights = torch.tensor(w, dtype=torch.float32)
        soft_subhead_weights = soft_subhead_weights / soft_subhead_weights.sum()
    if args.pos_weight_mode == "auto":
        pos_weight_sub = compute_pos_weight_sub_from_dataset(train_ds)
        if args.pos_weight_max is not None:
            pos_weight_sub = torch.clamp(pos_weight_sub, max=args.pos_weight_max)
        print("pos_weight_sub:", pos_weight_sub.tolist())


    # Build teacher map: seq_id -> logits tensor[5]
    t_train = load_teacher_logits(args.teacher_logits_train)
    t_val = load_teacher_logits(args.teacher_logits_val)
    t_all = pd.concat([t_train, t_val], axis=0)

    def row_to_tensor(row) -> torch.Tensor:
        z = [float(row[f"z_{name}"]) for name in LABEL_NAMES]
        return torch.tensor(z, dtype=torch.float32)

    teacher_map = {sid: row_to_tensor(t_all.loc[sid]) for sid in t_all.index.unique()}

    model = ESMForMultiLabel(args.student_model_dir)

    if args.grad_ckpt and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        fp16=bool(args.fp16),
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DistillTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        lambda_soft=args.lambda_soft,
        lambda_soft_any=args.lambda_soft_any,
        lambda_soft_sub=args.lambda_soft_sub,
        soft_subhead_weights=soft_subhead_weights,
        sampler_mode=args.sampler,
        sampler_max_mult=args.sampler_max_mult,
        sampler_power=args.sampler_power,
        teacher_map=teacher_map,
        alpha_sub=args.alpha_sub,
        pos_weight_sub=pos_weight_sub,
    )

    trainer.train()

    best_dir = os.path.join(args.output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    with open(os.path.join(args.output_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    metrics = trainer.evaluate()
    with open(os.path.join(args.output_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
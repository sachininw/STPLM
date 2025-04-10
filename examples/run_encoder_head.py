import os
import subprocess
import pickle
import datetime
from collections import Counter

import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments

from stplm import DataCollatorForCellClassification
from utils import STPLMForSequenceClassification, CustomTrainer


# --- Config ---
GPU_NUMBER = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPU_NUMBER))
os.environ["NCCL_DEBUG"] = "INFO"

DATA_PATH = "data/DATASETS_ORGAN/organ.dataset"
CHECKPOINT = "models/checkpoint/"
TRAIN_RATIO = 0.7

MAX_INPUT_SIZE = 2048
MAX_LR = 5e-5
FREEZE_LAYERS = 0
NUM_PROC = 16
BATCH_SIZE = 12
EPOCHS = 5
LR_SCHEDULER = "linear"
WARMUP_STEPS = 500
OPTIMIZER = "adamw"


def replace_non_cancer(example):
    example["cell_type"] = 'cancer' if example["cell_type"] == 'cancer' else 'non cancer'
    return example


def is_noncancer(example):
    return example["cell_type"] != 'cancer'


def filter_cell_types(dataset):
    counts = Counter(dataset["cell_type"])
    total = sum(counts.values())
    keep = [k for k, v in counts.items() if v > 0.00005 * total]
    return dataset.filter(lambda e: e["cell_type"] in keep, num_proc=NUM_PROC)


def encode_labels(dataset, label_map):
    return dataset.map(lambda e: {"label": label_map[e["label"]]}, num_proc=NUM_PROC)


def compute_metrics(pred):
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(pred.label_ids, preds),
        "macro_f1": f1_score(pred.label_ids, preds, average="macro")
    }


# --- Main Preprocessing and Training ---
def preprocess_and_train(train_dataset):
    organ_list = list(Counter(train_dataset["organ_major"]).keys())

    for organ in organ_list:
        print(f"\n=== Processing: {organ} ===")

        organ_ds = train_dataset.filter(lambda e: e["organ_major"] == organ, num_proc=NUM_PROC)
        organ_ds = organ_ds.filter(is_noncancer, num_proc=NUM_PROC)
        organ_ds = filter_cell_types(organ_ds)
        organ_ds = organ_ds.shuffle(seed=42)
        organ_ds = organ_ds.rename_column("cell_type", "label").remove_columns("organ_major")

        label_names = sorted(list(set(organ_ds["label"])))
        label_map = {name: i for i, name in enumerate(label_names)}
        print(f"Labels: {label_map}")

        organ_ds = encode_labels(organ_ds, label_map)
        train_size = int(len(organ_ds) * TRAIN_RATIO)
        train_ds = organ_ds.select(range(train_size))
        eval_ds = organ_ds.select(range(train_size, len(organ_ds)))
        eval_ds = eval_ds.filter(lambda e: e["label"] in set(train_ds["label"]), num_proc=NUM_PROC)

        train_and_save_model(train_ds, eval_ds, label_map, organ)


def train_and_save_model(train_ds, eval_ds, label_map, organ):
    logging_steps = max(1, len(train_ds) // BATCH_SIZE // 10)

    model = STPLMForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(label_map),
        output_attentions=False,
        output_hidden_states=False,
        ignore_mismatched_sizes=True
    ).to("cuda")

    datestamp = datetime.datetime.now().strftime("%y%m%d")
    output_dir = f"models/{datestamp}_STPLM_{organ}_L{MAX_INPUT_SIZE}_B{BATCH_SIZE}_LR{MAX_LR}_LS{LR_SCHEDULER}_WU{WARMUP_STEPS}_E{EPOCHS}_O{OPTIMIZER}_F{FREEZE_LAYERS}/"

    if os.path.exists(os.path.join(output_dir, "pytorch_model.bin")):
        raise FileExistsError(f"Model already exists in {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=MAX_LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.001,
        warmup_steps=WARMUP_STEPS,
        logging_steps=logging_steps,
        group_by_length=True,
        length_column_name="length",
        lr_scheduler_type=LR_SCHEDULER,
        fp16=True,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        save_total_limit=1,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        alpha_dataset=train_ds,
    )

    trainer.train()
    predictions = trainer.predict(eval_ds)

    with open(f"{output_dir}/predictions.pickle", "wb") as f:
        pickle.dump(predictions, f)

    trainer.save_metrics("eval", predictions.metrics)
    trainer.save_model(output_dir)


if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_from_disk(DATA_PATH)
    preprocess_and_train(dataset)

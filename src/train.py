import json
import time
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from datasets import Dataset

MODEL_NAME = "google/mt5-base"
DATA_DIR = Path("data/squad_bn")
OUTPUT_DIR = Path("outputs/model")
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4  # effective batch size stays 16
EPOCHS = 3
LEARNING_RATE = 1e-3  # mT5 + Adafactor needs a much higher LR than t5-base did


class EpochTimer(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        self._t0 = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self._t0
        epoch = int(state.epoch)
        print(f"\n  Epoch {epoch} time: {elapsed/60:.1f} min ({elapsed:.0f} s)")


def print_model_stats(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2
    print(f"  Trainable params : {trainable:,}")
    print(f"  Total params     : {total:,}")
    print(f"  Model size (fp32): {size_mb:.1f} MB")


def load_squad_json(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    ids, questions, contexts, answers = [], [], [], []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if not qa.get("answers"):
                    continue
                ids.append(qa["id"])
                questions.append(qa["question"])
                contexts.append(context)
                answers.append({
                    "text": [a["text"] for a in qa["answers"]],
                    "answer_start": [a["answer_start"] for a in qa["answers"]],
                })
    return Dataset.from_dict({"id": ids, "question": questions, "context": contexts, "answers": answers})


def tokenize(examples, tokenizer):
    inputs = [
        f"question: {q} context: {c}"
        for q, c in zip(examples["question"], examples["context"])
    ]
    targets = [
        a["text"][0] if a["text"] else ""
        for a in examples["answers"]
    ]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )
    labels["input_ids"] = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print("Model stats:")
    print_model_stats(model)

    print("Loading data...")
    train_ds = load_squad_json(DATA_DIR / "train.json")
    val_ds = load_squad_json(DATA_DIR / "validation.json")
    print(f"  train: {len(train_ds):,}  val: {len(val_ds):,}")

    print("Tokenizing...")
    train_features = train_ds.map(
        lambda ex: tokenize(ex, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
        desc="train",
    )
    val_features = val_ds.map(
        lambda ex: tokenize(ex, tokenizer),
        batched=True,
        remove_columns=val_ds.column_names,
        desc="val",
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        optim="adafactor",
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        bf16=use_bf16,
        report_to="none",
        logging_steps=200,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_features,
        eval_dataset=val_features,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EpochTimer()],
    )

    print("Training...")
    trainer.train()

    save_path = OUTPUT_DIR / "best"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()

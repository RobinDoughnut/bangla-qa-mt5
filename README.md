# Bangla QA with mT5-base

Fine-tuning [`google/mt5-base`](https://huggingface.co/google/mt5-base) for question answering on the Bangla SQuAD dataset ([csebuetnlp/squad_bn](https://huggingface.co/datasets/csebuetnlp/squad_bn)). Evaluated using Exact Match (EM), F1, and BERTScore-F1.

## Related Experiments

Part of a comparison of QA architectures on Bangla SQuAD:

| Repo | Model | Architecture |
|------|-------|--------------|
| [bangla-qa-banglat5](https://github.com/RobinDoughnut/bangla-qa-banglat5) | BanglaT5 | Encoder-Decoder, Bangla-pretrained |
| **bangla-qa-mt5** (this repo) | mT5-base | Encoder-Decoder, multilingual-pretrained |
| [bangla-t5-finetune-qa](https://github.com/RobinDoughnut/bangla-t5-finetune-qa) | T5-base | Encoder-Decoder, English-pretrained |
| [mbert-finetune-banglaSQUAD](https://github.com/RobinDoughnut/mbert-finetune-banglaSQUAD) | mBERT | Encoder-only, span extraction |

## Motivation

The T5-base experiment scored **EM 0.00 / F1 0.00** — not because the architecture is unsuited to the task, but because T5's English SentencePiece vocabulary has essentially no coverage of Bangla script. Every Bangla word tokenizes to `<unk>`:

```
input : শেখ মুজিবুর রহমান কবে জন্মগ্রহণ করেন ?
tokens: ['▁', '<unk>', '▁', '<unk>', '▁', '<unk>', '▁', '<unk>', '▁', '<unk>', '▁', '<unk>', '▁', '?', '</s>']
```

mT5 shares T5's architecture but is pretrained on mC4 across 101 languages with a 250k-token vocabulary that covers Bangla:

```
input : শেখ মুজিবুর রহমান কবে জন্মগ্রহণ করেন ?
unk   : 0/20 tokens — exact round-trip
```

This isolates **tokenizer and pretraining coverage** as the variable, holding architecture and input format constant, and sits between English-only T5 and Bangla-specific BanglaT5 on the pretraining-specificity axis.

| | T5-base | mT5-base | BanglaT5 |
|---|---|---|---|
| Pretrained on | English C4 | mC4 (101 languages) | Bangla corpus |
| Vocabulary | 32,100 | 250,100 | 32,000 |
| Bangla coverage | none | yes | yes |
| Parameters | 222,903,552 | 582,401,280 | 247,577,856 |
| Input format | `"question: Q context: C"` | `"question: Q context: C"` | `"question: Q context: C"` |

## Dataset

| Split | Examples |
|-------|----------|
| Train | 68,674 |
| Validation | 1,251 |
| Test | 1,252 |

(Unanswerable questions are excluded.)

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (~5.2 GB VRAM at the default batch size)

`protobuf` is required, not optional. mT5 ships only `spiece.model` with no prebuilt `tokenizer.json`; without `protobuf` installed, `transformers>=5` fails its SentencePiece backend check, silently falls back to the tiktoken converter, and dies with a misleading `` `tiktoken` is required `` error.

## Setup

```bash
git clone <repo-url>
cd bangla-qa-mt5

python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## Data

```bash
python src/prepare_data.py
```

Downloads `csebuetnlp/squad_bn` and saves splits under `data/squad_bn/`.

## Training

```bash
python src/train.py
```

Saves the best checkpoint to `outputs/model/best/`. Model stats and per-epoch timing are printed to stdout.

**Key hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Base model | `google/mt5-base` |
| Max input length | 512 tokens |
| Max target length | 64 tokens |
| Batch size (per device) | 4 |
| Gradient accumulation steps | 4 (effective batch = 16) |
| Epochs | 3 |
| Learning rate | 1e-3 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Optimizer | Adafactor |
| Mixed precision | bf16 (if supported) |
| Gradient checkpointing | Yes |

The learning rate is `1e-3`, not the `3e-5` used for T5-base. mT5 with Adafactor is conventionally fine-tuned around `1e-3`; at `3e-5` mT5-base tends to underfit substantially.

bf16 rather than fp16 matters here: mT5 is known to produce NaN losses under fp16.

## Evaluation

```bash
python src/evaluate_model.py
```

Generates predictions with beam search (4 beams) and reports EM, F1, and BERTScore-F1 on validation and test splits.

## Results

**Model stats:**

| Metric | Value |
|--------|-------|
| Trainable parameters | 582,401,280 |
| Total parameters | 582,401,280 |
| Model size (fp32) | 2221.7 MB |

**Training** (RTX 4070, effective batch size 16):

_Not yet run._

**Evaluation:**

_Not yet run._

## Project Structure

```
bangla-qa-mt5/
├── data/
│   └── squad_bn/          # gitignored — run prepare_data.py to populate
├── outputs/
│   └── model/             # gitignored — created during training
│       ├── best/
│       └── checkpoints/
├── src/
│   ├── prepare_data.py    # downloads dataset from HuggingFace
│   ├── train.py           # seq2seq fine-tuning script
│   └── evaluate_model.py  # generation + EM / F1 / BERTScore evaluation
├── requirements.txt
└── README.md
```

## Reproducing Results

```bash
git clone <repo-url> && cd bangla-qa-mt5
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/prepare_data.py
python src/train.py
python src/evaluate_model.py
```

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

## Data Preprocessing & Text Representation

Documented here for citation purposes.

**Preprocessing pipeline** (`src/prepare_data.py`, `src/train.py:load_squad_json`):

1. **Source**: `csebuetnlp/squad_bn` (Bangla SQuAD), loaded via the HuggingFace `datasets` library.
2. **Re-grouping**: the raw dataset is flat (one row per question-context pair, with the context
   string repeated for every question that shares it). Rows are re-grouped into SQuAD's canonical
   `{title: [{context, qas: [...]}]}` nested structure so each unique context appears once.
3. **Unanswerable-question filtering**: rows with an empty `answers` list are dropped. This is what
   shrinks the raw split sizes -- 118,117 / 2,502 / 2,504 train/validation/test in the current
   dataset snapshot -- down to 68,674 / 1,251 / 1,252, the sizes reported under Dataset above.
4. **Target answer selection**: `squad_bn` provides multiple human-annotated gold answer spans for
   some questions. Training uses only the first (`answers.text[0]`) as the target sequence `y`; at
   evaluation time all gold answers are retained, and EM/F1/BERTScore are computed against every
   gold answer with the maximum taken per example (standard SQuAD protocol).
5. **Prompt construction** (text-to-text framing, no task-specific output head):
   `"question: {question} context: {context}"` -> target: answer text.
6. **No text normalization** (lowercasing, punctuation stripping, diacritic folding) is applied
   before tokenization -- raw Bangla UTF-8 text is passed directly into the model's subword
   tokenizer.
7. **Truncation/padding**: input sequences truncated to 512 subword tokens, targets to 64; batches
   are dynamically padded per-batch (`DataCollatorForSeq2Seq`), and label padding positions are
   masked with `-100` so they don't contribute to the cross-entropy loss.

**Text embedding**:

- Tokenizer: SentencePiece unigram-model subword tokenizer (`T5Tokenizer`), operating directly on
  raw text -- distinct from the whitespace-then-WordPiece pipeline BERT-family models use.
- Vocabulary: 250,112 subword tokens, trained on **mC4** across 101 languages (multilingual,
  includes Bangla).
- Embedding dimension (`d_model`): 768 (mT5-**base**), across 12 encoder + 12 decoder layers.
- A single learned embedding matrix (`vocab_size x d_model`) maps token ids to dense vectors,
  **tied** (`tie_word_embeddings=True`) across the encoder input embedding, decoder input
  embedding, and output (LM head) projection in the pretrained checkpoint this repo starts from.
  Note the embedding matrix alone (250,112 x 768 ~= 192M parameters) is a large fraction of this
  model's 582M total parameters -- more than BanglaT5's entire parameter count.
- Positional information is **not** injected via absolute/sinusoidal embeddings; T5-family models
  add a learned **relative position bias** to the attention logits at every layer instead
  (`relative_attention_num_buckets=32`, `relative_attention_max_distance=128`).

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

| Epoch | Eval loss | Time |
|-------|-----------|------|
| 1 | 0.7784 | 61.6 min |
| 2 | 0.6870 | 61.9 min |
| 3 | 0.6535 | 61.6 min |

Total training runtime 3h05m at 18.5 samples/sec. Eval loss was still improving at epoch 3, so a longer run may gain further.

**Evaluation** (BERTScore-F1 uses `bert-base-multilingual-cased`, unrescaled, max over gold references):

| Split | EM | F1 | BERTScore-F1 | N |
|-------|-----|-----|--------------|-------|
| Validation | 39.97 | 56.24 | 87.07 | 1,251 |
| Test | 38.10 | 54.22 | 86.76 | 1,252 |

> **Finding:** swapping T5-base for mT5-base — same architecture, same input format, same data, same trainer — moves the score from **EM 0.00 / F1 0.00** to **EM 38.10 / F1 54.22** on test. The only thing that changed is vocabulary coverage of Bangla script. This confirms the T5-base result was a tokenizer failure rather than an architectural one.
>
> Note that T5-base reached a *lower* eval loss (0.366) than mT5-base (0.654) while scoring zero. Teacher-forced loss over degenerate all-`<unk>` targets is trivially minimized, so eval loss is not on its own evidence that a model works — generation-time metrics are required.
>
> mT5-base nonetheless finishes last among the models that can represent Bangla at all, despite being the largest:
>
> | Model | Test EM | Test F1 | BERTScore-F1 | Params |
> |-------|---------|---------|--------------|--------|
> | [BanglaT5](https://github.com/RobinDoughnut/bangla-qa-banglat5) | 53.19 | 67.90 | 91.10 | 248M |
> | [mBERT](https://github.com/RobinDoughnut/mbert-finetune-banglaSQUAD) | 52.24 | 65.87 | 90.18 | 178M |
> | **mT5-base** (this repo) | **38.10** | **54.22** | **86.76** | **582M** |
> | [T5-base](https://github.com/RobinDoughnut/bangla-t5-finetune-qa) | 0.00 | 0.00 | 1.65 | 223M |
>
> **This result should be treated as provisional.** Eval loss was still falling at epoch 3 (0.778 → 0.687 → 0.654) with no sign of overfitting, so 3 epochs likely under-trains mT5-base — its 250k-token embedding matrix has far more parameters to adapt than the other models, on the same 68,674 examples. The honest claim here is that mT5 clears the coverage threshold, not that it is settled as the weakest architecture. A longer run is needed before treating the gap to mBERT as real.

## Limitations

**Single run, no variance estimate.** Every number here comes from one training run at one seed. None of these repos report variance across seeds, so small gaps between models cannot be distinguished from run-to-run noise. This matters specifically for BanglaT5 (53.19 EM) vs mBERT (52.24 EM): a 0.95-point gap on 1,252 test examples is roughly 12 questions, and should not be read as a reliable ordering. The gap from either of those to mT5-base (38.10) is large enough to survive that objection; the gap between the two leaders is not.

**Epoch budget is not tuned per model.** All four experiments use 3 epochs, which favours models that converge quickly. mBERT had already begun overfitting by epoch 3 (its eval loss rose from 1.359 to 1.503), while mT5's was still falling. A fixed budget is a defensible controlled choice, but it is not the same as comparing each model at its own best configuration, and the results should not be described as "best achievable" for any of them.

**Dataset: `answer_start` offsets are unreliable in the eval splits.** In `csebuetnlp/squad_bn`, 167/1,251 validation answers (13.3%) and 181/1,332 test answers (13.6%) have an `answer_start` that does not land on the gold text — the pointer is 1–5 characters too large, most often by exactly 1. The gold **text** is intact in nearly all cases, and the `train` split is completely clean (0/78,328), so:

- Training is unaffected — span-extraction targets are built from the clean train split.
- The metrics reported here are unaffected — EM/F1/BERTScore compare against `answers["text"]`, never `answer_start`.
- Only 2 test examples (0.15%) have gold text that is itself clipped (`'৮৬২'` where the context reads `'১৮৬২'`), capping the distortion at roughly 0.16 EM.

The issue is documented because anyone reusing these splits for span-extraction *evaluation* — where predictions are scored against character offsets rather than text — would be materially affected, whereas this comparison is not.

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

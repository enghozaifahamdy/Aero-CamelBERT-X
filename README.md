# Aero-CamelBERT-X

Code and dataset for research on distinguishing human-written and AI-generated Arabic texts across research abstracts and reviews.

The project uses CAMeLBERT-MSA for binary classification and investigates drift-triggered adaptation and explanations using attention, SHAP, and LIME.

## Project scope

The research covers:

- Construction of a normalized, class-balanced Arabic corpus.
- Baseline classification using a frozen CAMeLBERT-MSA encoder and a trainable classification head.
- Evaluation across text domains and AI generators.
- Drift-triggered adaptation using ADWIN and EWC.
- Analysis of model explanations using attention, SHAP, and LIME.

The baseline notebook is provided. Adaptation and explanation components will be documented alongside their corresponding code.

## Dataset sources

The corpus is derived from two datasets released by KFUPM-JRCAI:

- [Arabic Generated Abstracts](https://huggingface.co/datasets/KFUPM-JRCAI/arabic-generated-abstracts)
- [Arabic Generated Social Media Posts](https://huggingface.co/datasets/KFUPM-JRCAI/arabic-generated-social-media-posts)

The research abstracts originate from ASJP. The review texts originate from BRAD and HARD.

AI texts were supplied by the source datasets using ALLaM, Jais, Llama, and OpenAI models. This project reuses those generations rather than generating the source texts independently.

Abstract generation methods include:

- Polishing an existing human abstract.
- Generation from a title.
- Generation from a title and content.

The review dataset uses polishing.

## Dataset construction

The construction notebook converts the source datasets into long format, with one text per row, then applies normalization, length filtering, deduplication, and class balancing.

### Arabic normalization

The preprocessing pipeline:

- Applies Unicode NFKC normalization.
- Removes tatweel and selected zero-width and directional characters.
- Normalizes Alef variants to ا, ى to ي, and ة to ه.
- Replaces control characters with spaces.
- Collapses repeated whitespace.
- Preserves Arabic diacritics.

### Length filtering

Research abstracts must contain at least 30 whitespace-separated words. Review texts must contain at least 50.

Word counts are computed after normalization using `len(text.split())`. These counts differ from model tokenizer tokens.

No entries were removed by these thresholds in the recorded construction run.

### Deduplication and balancing

Exact duplicates are removed using normalized text within each domain.

The three abstract generation-method splits contain repeated human abstracts. Consequently, their 8,388 human entries reduce to 2,992 retained human texts after deduplication.

All remaining human texts are retained. AI texts are sampled to match the human count within each domain, with approximately equal representation from the four generators. Sampling uses seed 42.

### Dataset construction summary

| Stage | Research abstracts | Reviews |
|---|---:|---:|
| Raw human entries | 8,388 | 3,318 |
| Raw AI entries | 33,552 | 13,272 |
| Raw total | 41,940 | 16,590 |
| After length filtering | 41,940 | 16,590 |
| After deduplication | 36,525 | 16,529 |
| Final human texts | 2,992 | 3,318 |
| Final AI texts | 2,992 | 3,318 |
| Final total | 5,984 | 6,636 |

The combined corpus contains **12,620 texts**, with **6,310 human texts** and **6,310 AI texts**.

### Dataset fields

| Field | Description |
|---|---|
| `text` | Normalized Arabic text |
| `label` | `human` or `ai` |
| `domain` | `research` or `social_media` |
| `source` | `ASJP` or `BRAD+HARD` |
| `llm` | `human`, `allam`, `jais`, `llama`, or `openai` |
| `method` | Source generation method; for human texts, the source method split |
| `n_words` | Number of whitespace-separated words after normalization |
| `split` | `train`, `val`, or `test` |

The internal domain value `social_media` refers to the BRAD/HARD review subset used in this project.

## Dataset splits

The corpus is divided into training, validation, and test partitions using stratification on the combined label and domain, with seed 42.

| Partition | Abstract human | Abstract AI | Review human | Review AI | Total |
|---|---:|---:|---:|---:|---:|
| Training | 2,394 | 2,394 | 2,654 | 2,654 | 10,096 |
| Validation | 299 | 299 | 332 | 332 | 1,262 |
| Test | 299 | 299 | 332 | 332 | 1,262 |
| Total | 2,992 | 2,992 | 3,318 | 3,318 | 12,620 |

### Known source-family overlap

The current partitions were created at the text-row level. Although the recorded train/test check found no identical normalized texts shared between those partitions, related human originals and AI variants can occur in different partitions.

The notebook's saved source-family diagnostic reports:

- 1,529 source families spanning multiple partitions.
- 845 of these families involving the test partition.

These are family counts, not counts of unique leaked texts. This overlap can affect evaluation and should be considered when interpreting results.

For an evaluation that separates related source items, construct source-grouped partitions that keep each human original and all associated AI variants together. Such partitions constitute a different evaluation protocol from the current released splits.

## Model and baseline training

The baseline uses:

- Encoder: `CAMeL-Lab/bert-base-arabic-camelbert-msa`
- Frozen encoder with a trainable classification head
- Binary labels: `human = 0`, `ai = 1`
- Maximum input length: 512 tokenizer tokens
- Batch size: 16
- Learning rate: 0.00002
- Dropout: 0.3
- Weight decay: 0.01
- Warmup ratio: 0.1
- Configured training epochs: 10
- Best-checkpoint selection by validation macro F1
- Random seed: 42

Inputs exceeding the tokenizer limit are truncated. Shorter inputs are padded.

The saved notebook outputs and loaded configuration specify 10 epochs; some introductory notebook text still refers to an earlier three-epoch setup.

## Running the notebooks

### 1. Build the corpus

Open the corpus-building notebook in Google Colab and run its construction cells in order.

The notebook downloads the source datasets from Hugging Face and exports:

- `arabic_ai_detection_corpus.csv`
- `arabic_ai_detection_corpus.parquet`
- `train.parquet`
- `val.parquet`
- `test.parquet`

The construction dependencies include `datasets`, `pandas`, `pyarrow`, and `scikit-learn`.

### 2. Configure baseline training

Open the Phase 2 baseline notebook in Google Colab with a GPU runtime.

The notebook expects a Google Drive project directory containing `configs/config.json`. Update the project directory and corpus path to match your environment.

The configuration must provide the training settings used by the notebook, including model name, maximum input length, batch size, learning rate, dropout, epochs, warmup ratio, weight decay, and seed.

### 3. Train and evaluate

Run the baseline notebook in order. It loads the existing dataset partitions, trains the classification head, selects the best checkpoint using validation macro F1, and evaluates that checkpoint on the test partition.

The notebook writes a model checkpoint, TensorBoard logs, and test metrics to the configured Google Drive project directory.

## Reproducibility

Use the released corpus and its existing split column to reproduce the reported partition sizes.

Rebuilding from the upstream datasets may produce different results if their contents change. Record source dataset revisions, package versions, configuration values, and random seeds when conducting new runs.

Inspect the executed configuration and saved outputs when reproducing experiments, because introductory notebook descriptions may not reflect every later configuration change.

## Interpretation and limitations

- The two domains differ in their generation-method composition, which can confound domain comparisons.
- Text-row splitting permits source-family overlap across partitions.
- Exact-text deduplication does not remove all paraphrases or near duplicates.
- Inputs longer than the model limit are truncated.
- Agreement between explanation methods does not, by itself, establish explanation faithfulness.
- AI-class recall should be distinguished from overall classification accuracy.
- Any adaptation experiment using correctness-based drift signals requires labels; its evaluation should document when labels become available and whether predictions precede updates.

This repository supports research and reproducibility. Model predictions should be interpreted in the context of the evaluation data and protocol.

## Attribution and licensing

Please cite the original source datasets and their associated publications when using this corpus.

Code licensing and dataset licensing are separate. Redistribution and reuse of the derived corpus remain subject to the applicable upstream licenses and terms. The repository's code license does not replace those conditions.

## Citation

The manuscript citation will be added when a public preprint or publication is available. Until then, acknowledge this repository and the original KFUPM-JRCAI datasets in work using these materials.

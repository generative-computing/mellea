# Mellea CLI — Train & Upload LoRA/aLoRA Adapters

Mellea provides a command-line interface for training and uploading [LoRA](https://arxiv.org/abs/2106.09685) or [aLoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora#alora) adapters for causal language models. This tool is useful for adapting base models like IBM Granite to custom tasks using prompt-based classification. The major goal is to help customer train a requirement validator.

---

## 🔧 Installation

From the root of the repository:

```bash
pip install mellea
huggingface-cli login  # Optional: only needed for uploads
```

---

## 📄 Training Data Format

Mellea expects training data in a `.jsonl` file, where each line contains:
- `item`: A user prompt or message
- `label`: A string classification label

### 📦 Example `data.jsonl`

```json
{"item": "The stembolt doesn't adjust at high RPM.", "label": "F"}
{"item": "Normal sensor readings but inconsistent throttle.", "label": "T"}
{"item": "Sluggish acceleration from idle.", "label": "T"}
```

---

## 🚀 Train a Model

Use the `m alora train` command to fine-tune a LoRA or aLoRA adapter requirement validator.

```bash
m alora train path/to/data.jsonl \
  --basemodel ibm-granite/granite-3.2-8b-instruct \
  --outfile ./checkpoints/alora_adapter \
  --adapter alora \
  --device auto \
  --epochs 6 \
  --learning-rate 6e-6 \
  --batch-size 2 \
  --max-length 1024 \
  --grad-accum 4
```

### 📌 Parameters

| Flag              | Type    | Default   | Description                                      |
|-------------------|---------|-----------|--------------------------------------------------|
| `--basemodel`     | `str`   | *required*| Hugging Face model ID or local path              |
| `--outfile`       | `str`   | *required*| Directory to save the adapter weights            |
| `--adapter`       | `str`   | `"alora"` | Choose between `alora` or standard `lora`        |
| `--device`        | `str`   | `"auto"`  | Device: `auto`, `cpu`, `cuda`, or `mps`          |
| `--epochs`        | `int`   | `6`       | Number of training epochs                        |
| `--learning-rate` | `float` | `6e-6`    | Learning rate                                    |
| `--batch-size`    | `int`   | `2`       | Per-device batch size                            |
| `--max-length`    | `int`   | `1024`    | Max tokenized input length                       |
| `--grad-accum`    | `int`   | `4`       | Gradient accumulation steps                      |

---

## ⬆️ Upload to Hugging Face

Use the `m alora upload` command to publish your trained adapter:

```bash
m alora upload ./checkpoints/alora_adapter \
  --name acme/carbchecker-alora
```

This will:
- Create the Hugging Face model repo (if it doesn't exist)
- Upload the contents of the `outfile` directory
- Requires a valid `HF_TOKEN` via `huggingface-cli login`

---


## 🛠 Requirements

- Python 3.8+
- Install the following dependencies manually or via `pip install mellea[hf]`:
  - `transformers`
  - `trl`
  - `peft>=0.18.1` (native aLoRA support)
  - `datasets`
  - `huggingface_hub`


---

## 🧪 Example Datasets for Testing

To verify the `alora-train` and `alora-upload` functionality, we tested the CLI using two well-known benchmark datasets: **TREC** and **SST-2**. These datasets are small, well-structured, and suitable for validating training pipelines.

### 📚 1. TREC (Question Classification)

- **Link**: [Hugging Face: TREC Dataset](https://huggingface.co/datasets/trec)
- **Description**: The TREC dataset consists of open-domain, fact-based questions divided into broad semantic categories. Each example contains a question and a label such as `DESC`, `HUM`, `LOC`, etc.
- **Used format**:
  ```json
  {"item": "What is the capital of France?", "label": "LOC"}


### 📚 2. SST-2 (Stanford Sentiment Treebank v2)

- **Link**: [Hugging Face: sst-2 Dataset](https://huggingface.co/datasets/stanfordnlp/sst2)
- **Description**: SST-2 is a binary sentiment classification dataset based on movie review sentences. Each entry is labeled as either `POSITIVE` or `NEGATIVE`.
- **Used format**:
  ```json
  {"item": "A beautiful, poetic piece of cinema.", "label": "POSITIVE"}

---
base_model: meta-llama/Llama-3.2-1B-Instruct
library_name: peft
---

# Persona-Driven Review & Rating Predictor

This project contains the code for a fine-tuned  model designed to predict user ratings and text reviews based on user personas and item identifiers.

---

## Training Data & Inputs

The model was fine-tuned on a merged dataset consisting of product reviews (amazon review) and generated user persona profiles (the user persona profile was generated with the aid of an LLM, using the user information from amazon review database). see [here](https://amazon-reviews-2023.github.io/) for dataset.

*   **Input Context:** 
    *   `persona`: A structured JSON object containing a user's values, writing style markers, common complaints, and rating distribution.
    *   `title`: The textual headline of the product being review.
*   **Target Outputs:**
    *   `average_rating`: Numerical target (1.0 to 5.0).
    *   `text`: The detailed review body reflecting the persona's tone, detail level, and vocabulary markers.

---

- Link to training data set: https://huggingface.co/datasets/ChisomChibuike/amazon-review-persona/tree/main

- Link to model: https://huggingface.co/ChisomChibuike/amazon-rating-sentiment

## Fine-Tuning

Fine-tune **Llama 3.2-1B** on reviews using **QLoRA** (4-bit quantization + LoRA adapters) on **Modal** GPU infrastructure.

**Task:** Given a user persona + product description, return the predict star rating (1–5) + write a review.

---

## Project Structure

```
Instruction-tuning/
│
├── main.py              
├── modal_img.py         
├── data_prep.py        
├── final_reviews.csv    
│
├── train/
│   ├── app.py           
│   ├── training.py      
│   ├── chatml.py        
│   └── utils.py        
│
└── evaluate/
    ├── app.py           
    ├── evaluate.py      
    ├── load_model.py    
    └── evalml.py        
```

---

## Prerequisites

### 1. Install Modal locally

```bash
pip install modal
```

### 2. Authenticate with Modal

```bash
modal setup
```

This opens a browser — sign in and your token is saved automatically.

### 3. Store your secrets

```bash
# HuggingFace token (needs Llama 3.2 access approved at huggingface.co/meta-llama)
modal secret create huggingface HF_TOKEN=hf_xxxxxxxxxxxxxx

# Weights & Biases API key (from wandb.ai/settings)
modal secret create wandb WANDB_API_KEY=xxxxxxxxxxxxxx
```

### 4. Accept Llama 3.2 license

Go to [huggingface.co/meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and accept the license with the same HuggingFace account your token belongs to.

### 5. Place your dataset

Make sure `final_reviews.csv` is in the root of the project folder.

---

## Running the Pipeline

### Step 1 — Data Prep + Training (full pipeline)

```powershell
modal run -m main::main_train
```

This will:
1. Read `final_reviews.csv` from your local machine
2. Clean text (strip HTML, remove emojis, decode entities)
3. Format into ChatML instruction examples
4. Split 90% train / 10% val and save to Modal volume
5. Fine-tune Llama 3.2-1B with QLoRA on an A100 GPU
6. Save final weights to the Modal volume

Training logs stream live to your terminal and to your W&B dashboard.

---

### Step 2 — Evaluation

```powershell
modal run -m main::main_eval
```

This will:
1. Load the fine-tuned model from the Modal volume
2. Run inference on val examples.
3. Report metrics: Rating MAE, Exact Accuracy, Acc±1, ROUGE-1
4. Log a full predictions table to W&B
5. Save `eval_results.csv` and `metrics.json` to the Modal volume

---

### Download results to your machine

```powershell
# Download final model
modal volume get llama-finetune-vol /final_model ./model

# Download evaluation results
modal volume get llama-finetune-vol /eval_results ./eval_results
```

---

## Evaluation Metrics

| Metric | Description | Good value |
|---|---|---|
| `eval/rating_mae` | Mean absolute star error | < 0.8 |
| `eval/rating_acc_exact` | % with perfectly correct star | > 40% |
| `eval/rating_acc_within1` | % correct within 1 star | > 75% |
| `eval/avg_rouge1` | Word overlap with ground-truth review | > 0.25 |

---

## Model & Training Config

| Setting | Value |
|---|---|
| Base model | `meta-llama/Llama-3.2-1B-Instruct` |
| Quantization | 4-bit NF4 (bitsandbytes) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA target modules | q_proj, v_proj, k_proj, o_proj, gate_proj |
| Epochs | set to 100 but early stopping implemented |
| Batch size | 4 × 4 grad accumulation = effective 16 |
| Learning rate | 2e-4 cosine with warmup |
| GPU | A100-40GB (training) / A10G (evaluation) |
| Max sequence length | 512 tokens |

---


**More data** — Adding more diverse personas and products will improve generalisation for the model performance.

---

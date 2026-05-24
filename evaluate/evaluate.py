import modal
import os, json
import pandas as pd
import wandb
from datasets import load_from_disk
from .app import app, VOL, volume, DATA_DIR, MODEL_DIR, RESULT_DIR
from .load_model import load_model_and_tokenizer, generate
from rouge_score import rouge_scorer
from .evalml import parse_output



@app.function(
    gpu="A10G",               
    volumes={VOL: volume},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
    timeout=3600,
)


def run_eval(num_samples: int = 100):
    """
    Evaluate the fine-tuned model on the validation set.
 
    Args:
        num_samples: how many val examples to evaluate.
                     Pass -1 to evaluate the full val set.
    """
    volume.reload()
 
    val_ds = load_from_disk(f"{DATA_DIR}/val")
    if num_samples > 0:
        val_ds = val_ds.select(range(min(num_samples, len(val_ds))))
    
    model, tokenizer = load_model_and_tokenizer(MODEL_DIR)
 
    wandb.init(
        project="llama-review-finetune",
        name="eval-run",
        job_type="evaluation",
    )
    table = wandb.Table(columns=[
        "product", "true_rating", "pred_rating",
        "rating_error", "pred_review_snippet", "rouge1"
    ])
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
 
    results = []
    for i, ex in enumerate(val_ds):
        full_text  = ex["text"]
        split_tag  = "<|start_header_id|>assistant<|end_header_id|>\n"
        if split_tag in full_text:
            prompt_part   = full_text.split(split_tag)[0] + split_tag
            gold_response = full_text.split(split_tag)[1].replace("<|eot_id|>", "").strip()
        else:
            prompt_part   = full_text
            gold_response = ""
 
        true_rating, true_review = parse_output(gold_response)
 
        raw_output   = generate(model, tokenizer, prompt_part)
        pred_rating, pred_review = parse_output(raw_output)
 
        rating_error = (
            abs(pred_rating - true_rating)
            if pred_rating is not None and true_rating is not None
            else None
        )
 
        rouge1 = 0.0
        if pred_review and true_review:
            rouge1 = scorer.score(true_review, pred_review)["rouge1"].fmeasure
 
        results.append({
            "true_rating":  true_rating,
            "pred_rating":  pred_rating,
            "rating_error": rating_error,
            "pred_review":  pred_review,
            "true_review":  true_review,
            "rouge1":       rouge1,
        })
 
        table.add_data(
            f"example_{i}",
            true_rating,
            pred_rating,
            rating_error,
            (pred_review or "")[:200],
            round(rouge1, 3),
        )
 
    df = pd.DataFrame(results)
 
    valid    = df.dropna(subset=["true_rating", "pred_rating", "rating_error"])
    mae      = valid["rating_error"].mean()
    acc_exact = (valid["rating_error"] == 0).mean()
    acc_1    = (valid["rating_error"] <= 1).mean()
    avg_rouge1 = df["rouge1"].mean()
 
    metrics = {
        "eval/rating_mae":        round(mae, 4),
        "eval/rating_acc_exact":  round(acc_exact, 4),
        "eval/rating_acc_within1":round(acc_1, 4),
        "eval/avg_rouge1":        round(avg_rouge1, 4),
        "eval/n_samples":         len(df),
        "eval/n_parsed":          len(valid),
    }
 
    wandb.log(metrics)
    wandb.log({"eval/predictions": table})
    wandb.finish()
    
    os.makedirs(RESULT_DIR, exist_ok=True)
    df.to_csv(f"{RESULT_DIR}/eval_results.csv", index=False)
    with open(f"{RESULT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    volume.commit()
 
    return metrics
 
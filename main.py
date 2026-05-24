import os
import json
from train.app import app as train_app
from evaluate.app import app as eval_app
from data_prep import prep_data
from train.training import train
from evaluate.evaluate import run_eval

@train_app.local_entrypoint()
def main_train():
    csv_path = "final_reviews.csv"    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find '{csv_path}'. "
        )
    with open(csv_path, "r", encoding="utf-8") as f:
        csv_content = f.read()
    n_train, n_val = prep_data.remote(csv_content)
 
    train.remote()
    print("Done, weights saved to Modal volume")
    
    
@eval_app.local_entrypoint()
def main_eval():
    metrics = run_eval.remote(num_samples=100)
    print(json.dumps(metrics, indent=2))
    print("\nResults saved to Modal volume.")
import modal
from modal_img import image

app    = modal.App("llama-review-eval", image=image)
volume = modal.Volume.from_name("llama-finetune-vol", create_if_missing=True)
 
VOL        = "/vol"
DATA_DIR   = f"{VOL}/data"
MODEL_DIR  = f"{VOL}/final_model"
RESULT_DIR = f"{VOL}/eval_results"

import modal
from modal_img import image

app    = modal.App("llama-review-finetune", image=image)
volume = modal.Volume.from_name("llama-finetune-vol", create_if_missing=True)
 
VOL          = "/vol"
DATA_DIR     = f"{VOL}/data"
CKPT_DIR     = f"{VOL}/checkpoints"
OUTPUT_DIR   = f"{VOL}/final_model"
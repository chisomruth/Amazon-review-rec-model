import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "transformers==4.44.2",
        "trl==0.9.6",
        "peft==0.12.0",
        "bitsandbytes==0.43.3",
        "datasets==2.20.0",
        "accelerate==0.33.0",
        "scikit-learn==1.5.1",
        "pandas==2.2.2",
        "wandb==0.17.5",
        "huggingface_hub==0.24.5",
        "rouge-score==0.1.2",
        "rich",
    )
     .add_local_python_source("train", "evaluate", "data_prep", "modal_img")
)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_model_and_tokenizer(model_dir: str):
 
    base_id  = "meta-llama/Llama-3.2-1B-Instruct"
    hf_token = __import__("os").environ.get("HF_TOKEN")
 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
 
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    return model, tokenizer
 
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 300) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
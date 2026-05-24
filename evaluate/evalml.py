import re

def build_prompt(persona_text: str, title: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are an expert product reviewer. "
        "Given a user persona and a product description, predict how that user "
        "would rate the product (1–5 stars) and write their review.\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"User persona:\n{persona_text}\n\n"
        f"Product description: {title}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    
    
def parse_output(raw: str):
    """
    Extract rating (int) and review (str) from model output.
    Expected format:
        Rating: 4/5\n\nReview:\n<text>
    """
    rating, review = None, raw.strip()
 
    m = re.search(r"[Rr]ating[:\s]+([1-5])", raw)
    if m:
        rating = int(m.group(1))
 
    m2 = re.search(r"[Rr]eview[:\s]*\n?(.*)", raw, re.DOTALL)
    if m2:
        review = m2.group(1).strip()
 
    return rating, review
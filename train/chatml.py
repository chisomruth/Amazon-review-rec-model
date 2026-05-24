def to_chatml(persona_text: str, title: str, rating: int, review: str) -> dict:
    """ChatML training."""
    return {
        "text": (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            "You are an expert product reviewer. "
            "Given a user persona, and product description, predict how that user "
            "would rate the product (1–5 stars) and write their review.\n"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"User persona:\n{persona_text}\n\n"
            f"Product description: {title}\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            f"Rating: {rating}/5\n\nReview:\n{review}\n"
            "<|eot_id|>"
        )
    }
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

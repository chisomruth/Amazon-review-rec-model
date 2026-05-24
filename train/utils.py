def remove_emojis(text: str) -> str:
    import unicodedata
    return "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("So")
    )
 
def clean(text: str) -> str:
    import re
    text = str(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<.*?>", "", text)                          
    text = (text
            .replace("&amp;", "&").replace("&lt;", "<")
            .replace("&gt;", ">").replace("&#39;", "'")
            .replace("&quot;", '"'))
    text = remove_emojis(text)
    text = re.sub(r"\n{3,}", "\n\n", text)   
    text = re.sub(r" {2,}", " ", text).strip()
    return text[:1024]
 
 
def format_persona(raw: str) -> str:
    """
    The persona column is a JSON string.
    We pull out the most useful plain-text fields for the prompt.
    Falls back to the raw string if parsing fails.
    """
    import json, re
    try:
        p = json.loads(raw)
        prefs = p.get("preferences", {})
        style = p.get("writing_style", {})
        rated = p.get("rating_behavior", {})
 
        lines = []
        if prefs.get("what_they_value"):
            lines.append("Values: " + ", ".join(prefs["what_they_value"]))
        if prefs.get("liked_attributes"):
            lines.append("Likes: " + ", ".join(prefs["liked_attributes"]))
        if prefs.get("disliked_attributes"):
            lines.append("Dislikes: " + ", ".join(prefs["disliked_attributes"]))
        if style.get("tone"):
            lines.append("Writing tone: " + style["tone"])
        if rated.get("rating_patterns"):
            lines.append("Rating style: " + rated["rating_patterns"])
 
        return "\n".join(lines) if lines else raw[:512]
    except Exception:
        return re.sub(r'[{}"\\]', ' ', raw)[:512].strip()
 
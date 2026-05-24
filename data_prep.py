import io, os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from train.app import app, volume, VOL, DATA_DIR
from modal_img import image
from train.utils import clean, format_persona
from train.chatml import to_chatml


@app.function(
    volumes={VOL: volume},
    timeout=600,
)

def prep_data(csv_content: str):

    df = pd.read_csv(io.StringIO(csv_content))
    required = ["persona", "average_rating", "title", "text"]
    df = df[required].dropna(subset=required).reset_index(drop=True)
    
    records = []
    for _, row in df.iterrows():
        persona_text = format_persona(str(row["persona"]))
        title        = clean(str(row["title"]))
        rating       = int(float(row["average_rating"])) 
        review       = clean(str(row["text"]))
        records.append(to_chatml(persona_text, title, rating, review))
 
    train_records, val_records = train_test_split(
        records, test_size=0.1, random_state=42
    )
 
    os.makedirs(DATA_DIR, exist_ok=True)
    Dataset.from_list(train_records).save_to_disk(f"{DATA_DIR}/train")
    Dataset.from_list(val_records).save_to_disk(f"{DATA_DIR}/val")
    volume.commit()
 
    return len(train_records), len(val_records)
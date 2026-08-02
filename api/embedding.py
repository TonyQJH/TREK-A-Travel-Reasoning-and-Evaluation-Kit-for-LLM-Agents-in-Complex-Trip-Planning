import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# Resolved relative to this file so the script runs from any checkout; override
# with TREK_HOTEL_DATA_DIR if the knowledge base lives elsewhere. (The previous
# hard-coded path also pointed at a pre-v2 layout that no longer exists.)
_HERE = os.path.dirname(os.path.abspath(__file__))
input_dir = os.environ.get("TREK_HOTEL_DATA_DIR",
                           os.path.join(_HERE, "data", "v2", "hotel_data"))
output_dir = os.path.join(input_dir, "amenities_group_embedding")
os.makedirs(output_dir, exist_ok=True)

csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
total = len(csv_files)

for idx, fname in enumerate(csv_files):
    city = fname.replace('_hotel.csv', '')
    df = pd.read_csv(os.path.join(input_dir, fname))
    # 注意这里是 amenities_group
    group_strs = [", ".join(eval(x)) for x in df["amenities_group"]]
    embeds = model.encode(group_strs, prompt_name="query")
    np.save(os.path.join(output_dir, f"{city}_amenities_group.npy"), embeds)
    print(f"Processing {city} ({idx+1}/{total})")  # 进度打印

print("全部城市 embedding 已完成！")

"""Oxford-IIIT Pet veri setini Hugging Face'ten indirir (data/data/*.parquet)."""
from huggingface_hub import hf_hub_download
for s in ["train", "test"]:
    hf_hub_download("timm/oxford-iiit-pet", f"data/{s}-00000-of-00001.parquet",
                    repo_type="dataset", local_dir="data")
print("Veri seti indirildi: data/data/")

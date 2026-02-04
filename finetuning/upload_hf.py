from huggingface_hub import upload_folder, login
import pyarrow.parquet as pq
from pathlib import Path

# (optional) Login with your Hugging Face credentials
login()

# Split parquet into chunks
input_path = Path("./sft_dataset/merged.parquet")
output_dir = Path("./sft_dataset/chunks")
output_dir.mkdir(parents=True, exist_ok=True)

table = pq.read_table(input_path)
chunk_size = 100_000  # rows per chunk

for i, start in enumerate(range(0, len(table), chunk_size)):
    chunk = table.slice(start, chunk_size)
    pq.write_table(chunk, output_dir / f"part_{i:04d}.parquet")

print(f"Split into {(len(table) + chunk_size - 1) // chunk_size} chunks")

# Upload folder
upload_folder(
    folder_path=str(output_dir),
    path_in_repo="data",
    repo_id="HyperBlaze/Qwen3TTS-game-sft",
    repo_type="dataset",
)


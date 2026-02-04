from huggingface_hub import upload_folder, login
import pyarrow.parquet as pq
from pathlib import Path

# Split parquet into chunks (streaming to avoid OOM)
input_path = Path("./sft_dataset/merged.parquet")
output_dir = Path("./sft_dataset/chunks")
output_dir.mkdir(parents=True, exist_ok=True)

parquet_file = pq.ParquetFile(input_path)
chunk_size = 100_000  # rows per batch

for i, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
    pq.write_table(pq.Table.from_batches([batch]), output_dir / f"part_{i:04d}.parquet")
    print(f"Wrote chunk {i}")

print(f"Split complete")

# Upload folder
upload_folder(
    folder_path=str(output_dir),
    path_in_repo="data",
    repo_id="HyperBlaze/Qwen3TTS-game-sft",
    repo_type="dataset",
)


# Example usage:
#   python sft_12hz_hf.py \
#       --hf_dataset HyperBlaze/Qwen3TTS-game-sft \
#       --speaker_name game_voice \
#       --output_model_path output/game_voice
import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from datasets import Audio, load_dataset
from dataset_hf import HFTTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-TTS on a HuggingFace dataset with pre-computed audio codes."
    )
    # Model paths
    parser.add_argument(
        "--init_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Path or HF repo of the base model.",
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        default="output",
        help="Directory to save checkpoints.",
    )

    # Dataset arguments
    parser.add_argument(
        "--hf_dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name or path (e.g., HyperBlaze/Qwen3TTS-game-sft).",
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default="train",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace datasets.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large datasets (not recommended for training).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples (for debugging).",
    )

    # Column names (for flexibility with different dataset schemas)
    parser.add_argument(
        "--audio_column",
        type=str,
        default="audio",
        help="Column name for audio data.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column name for transcription text.",
    )
    parser.add_argument(
        "--codes_column",
        type=str,
        default="audio_codes",
        help="Column name for pre-computed audio codes.",
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0)

    # Speaker configuration
    parser.add_argument(
        "--speaker_name",
        type=str,
        default="speaker_test",
        help="Name for the fine-tuned speaker voice.",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=3000,
        help="Speaker ID slot to use in the embedding table.",
    )

    # Logging
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Log loss every N steps.",
    )
    parser.add_argument(
        "--save_every_epoch",
        action="store_true",
        default=True,
        help="Save checkpoint after each epoch.",
    )

    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
    )

    accelerator.print("=" * 60)
    accelerator.print("Qwen3-TTS Fine-tuning with HuggingFace Dataset")
    accelerator.print("=" * 60)
    accelerator.print(f"  Base model: {args.init_model_path}")
    accelerator.print(f"  Dataset: {args.hf_dataset} (split={args.hf_split})")
    accelerator.print(f"  Speaker name: {args.speaker_name} (id={args.speaker_id})")
    accelerator.print(f"  Batch size: {args.batch_size}")
    accelerator.print(f"  Learning rate: {args.lr}")
    accelerator.print(f"  Epochs: {args.num_epochs}")
    accelerator.print("=" * 60)

    MODEL_PATH = args.init_model_path

    accelerator.print("Loading model...")
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    accelerator.print(f"Loading dataset: {args.hf_dataset}...")
    hf_dataset = load_dataset(
        args.hf_dataset,
        split=args.hf_split,
        streaming=args.streaming,
        cache_dir=args.hf_cache_dir,
    )

    # Don't decode audio - we'll handle it in the dataset class
    if hasattr(hf_dataset, "features") and args.audio_column in hf_dataset.features:
        try:
            hf_dataset = hf_dataset.cast_column(args.audio_column, Audio(decode=False))
        except Exception:
            pass

    if args.max_samples is not None:
        accelerator.print(f"Limiting to {args.max_samples} samples")
        if args.streaming:
            hf_dataset = hf_dataset.take(args.max_samples)
        else:
            hf_dataset = hf_dataset.select(range(min(args.max_samples, len(hf_dataset))))

    # For streaming datasets, we need to convert to a list or use IterableDataset
    if args.streaming:
        accelerator.print("[warn] Streaming mode converts to list - may use significant memory")
        hf_dataset = list(hf_dataset)

    dataset = HFTTSDataset(
        hf_dataset,
        qwen3tts.processor,
        config,
        audio_column=args.audio_column,
        text_column=args.text_column,
        codes_column=args.codes_column,
    )

    accelerator.print(f"Dataset size: {len(dataset)} samples")

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=0,  # Audio processing doesn't benefit much from workers
    )

    optimizer = AdamW(
        qwen3tts.model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()

    accelerator.print("Starting training...")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]

                # Extract speaker embedding from reference mel
                speaker_embedding = model.speaker_encoder(
                    ref_mels.to(model.device).to(model.dtype)
                ).detach()

                # Store first speaker embedding as target (for saving)
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = (
                    model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                )
                input_codec_embedding = (
                    model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                )
                # Inject speaker embedding at position 6
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                # Add codec embeddings for layers 1-15
                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](
                        codec_ids[:, :, i]
                    )
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1

            if step % args.log_every == 0:
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | "
                    f"Main: {outputs.loss.item():.4f} | Sub: {sub_talker_loss.item():.4f}"
                )

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        accelerator.print(f"Epoch {epoch} completed | Average Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        if accelerator.is_main_process and args.save_every_epoch:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            accelerator.print(f"Saving checkpoint to {output_dir}...")

            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            # Update config for custom voice
            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {args.speaker_name: args.speaker_id}
            talker_config["spk_is_dialect"] = {args.speaker_name: False}
            config_dict["talker_config"] = talker_config

            with open(output_config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            # Save model weights
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            # Drop speaker encoder weights (not needed for inference)
            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            # Inject speaker embedding into codec embedding table
            weight = state_dict["talker.model.codec_embedding.weight"]
            state_dict["talker.model.codec_embedding.weight"][args.speaker_id] = (
                target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            )

            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)
            accelerator.print(f"Checkpoint saved: {output_dir}")

    accelerator.print("Training complete!")


if __name__ == "__main__":
    train()

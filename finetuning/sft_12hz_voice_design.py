# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Voice Design SFT (12Hz).

Expected JSONL format (one JSON per line):
  {"audio":"./data/utt0001.wav","text":"...","instruct":"Bright, youthful female voice."}

After prepare_data.py, each line should also include:
  "audio_codes": [[...], ...]
"""
import argparse
import json
import os
import shutil
from typing import Optional

import torch
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


class VoiceDesignTTSDataset(Dataset):
    def __init__(self, data_list, processor, config: Qwen3TTSConfig, lag_num: int = -1):
        self.data_list = data_list
        self.processor = processor
        self.lag_num = lag_num
        self.config = config

    def __len__(self):
        return len(self.data_list)

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_instruct_text(self, instruct: str) -> str:
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def _tokenize_texts(self, text) -> torch.Tensor:
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        return input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids

    def __getitem__(self, idx):
        item = self.data_list[idx]

        text = item["text"]
        if "instruct" not in item:
            raise KeyError("Missing required field 'instruct' for voice design training.")
        instruct = item["instruct"]
        audio_codes = item["audio_codes"]

        text = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text)

        instruct_ids: Optional[torch.Tensor]
        if instruct is None or instruct == "":
            instruct_ids = None
        else:
            instruct_text = self._build_instruct_text(instruct)
            instruct_ids = self._tokenize_texts(instruct_text)

        audio_codes = torch.tensor(audio_codes, dtype=torch.long)

        return {
            "text_ids": text_ids[:, :-5],  # 1, t
            "audio_codes": audio_codes,    # t, 16
            "instruct_ids": instruct_ids,  # 1, t_ins or None
        }

    def collate_fn(self, batch):
        assert self.lag_num == -1

        item_length = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
        max_length = max(item_length) + 8
        bsz, seq_len = len(batch), max_length

        input_ids = torch.zeros((bsz, seq_len, 2), dtype=torch.long)
        codec_ids = torch.zeros((bsz, seq_len, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((bsz, seq_len), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((bsz, seq_len), dtype=torch.bool)
        codec_mask = torch.zeros((bsz, seq_len), dtype=torch.bool)
        attention_mask = torch.zeros((bsz, seq_len), dtype=torch.long)
        codec_0_labels = torch.full((bsz, seq_len), -100, dtype=torch.long)

        for i, data in enumerate(batch):
            text_ids = data["text_ids"]
            audio_codec_0 = data["audio_codes"][:, 0]
            audio_codecs = data["audio_codes"]

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            # text channel
            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8:8 + text_ids_len - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + text_ids_len - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8 + text_ids_len - 2:8 + text_ids_len + codec_ids_len, 0] = self.config.tts_pad_token_id
            text_embedding_mask[i, :8 + text_ids_len + codec_ids_len] = True

            # codec channel
            input_ids[i, 3:8, 1] = torch.tensor(
                [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    0,  # speaker embedding slot (unused in voice design)
                    self.config.talker_config.codec_pad_id,
                ]
            )
            input_ids[i, 8:8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 2, 1] = self.config.talker_config.codec_bos_id
            input_ids[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len, 1] = audio_codec_0
            input_ids[i, 8 + text_ids_len - 1 + codec_ids_len, 1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len] = audio_codec_0
            codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = self.config.talker_config.codec_eos_token_id

            codec_ids[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len, :] = audio_codecs

            codec_embedding_mask[i, 3:8 + text_ids_len + codec_ids_len] = True
            codec_embedding_mask[i, 6] = False  # speaker slot unused in voice design

            codec_mask[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len] = True
            attention_mask[i, :8 + text_ids_len + codec_ids_len] = True

        # instruct ids padding
        instruct_ids_list = [b["instruct_ids"] for b in batch]
        instruct_lengths = [ids.shape[1] if ids is not None else 0 for ids in instruct_ids_list]
        max_instruct_len = max(instruct_lengths) if instruct_lengths else 0

        instruct_ids = torch.zeros((bsz, max_instruct_len), dtype=torch.long)
        instruct_mask = torch.zeros((bsz, max_instruct_len), dtype=torch.bool)
        for i, ids in enumerate(instruct_ids_list):
            if ids is None or ids.numel() == 0:
                continue
            ids = ids.squeeze(0)
            length = ids.shape[0]
            instruct_ids[i, :length] = ids
            instruct_mask[i, :length] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
            "instruct_ids": instruct_ids,
            "instruct_mask": instruct_mask,
        }


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")

    model_path = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(model_path)

    train_data = [json.loads(line) for line in open(args.train_jsonl).readlines()]
    dataset = VoiceDesignTTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    model.train()

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]
                instruct_ids = batch["instruct_ids"]
                instruct_mask = batch["instruct_mask"]

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask

                # match inference behavior: no speaker embedding, use codec_pad at speaker slot
                pad_id = config.talker_config.codec_pad_id
                pad_embed = model.talker.model.codec_embedding.weight[pad_id]
                input_codec_embedding[:, 6, :] = pad_embed

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                if instruct_ids is not None and instruct_ids.shape[1] > 0:
                    instruct_embeddings = model.talker.text_projection(
                        model.talker.get_text_embeddings()(instruct_ids)
                    )
                    instruct_embeddings = instruct_embeddings * instruct_mask.unsqueeze(-1)

                    input_embeddings = torch.cat([instruct_embeddings, input_embeddings], dim=1)

                    prefix_len = instruct_ids.shape[1]
                    attention_mask = torch.cat(
                        [instruct_mask.to(attention_mask.dtype), attention_mask], dim=1
                    )
                    codec_mask = torch.cat(
                        [torch.zeros((codec_mask.shape[0], prefix_len), dtype=torch.bool, device=codec_mask.device), codec_mask],
                        dim=1,
                    )
                    codec_0_labels = torch.cat(
                        [torch.full((codec_0_labels.shape[0], prefix_len), -100, dtype=codec_0_labels.dtype, device=codec_0_labels.device), codec_0_labels],
                        dim=1,
                    )
                    codec_ids = torch.cat(
                        [torch.zeros((codec_ids.shape[0], prefix_len, codec_ids.shape[2]), dtype=codec_ids.dtype, device=codec_ids.device), codec_ids],
                        dim=1,
                    )

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                _, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(model_path, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(model_path, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "voice_design"

            with open(output_config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)


if __name__ == "__main__":
    train()

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
Voice Design DPO (12Hz).

Expected JSONL format (one JSON per line):
  {
    "text": "...",
    "instruct": "Bright, youthful female voice.",
    "audio_codes": [[...], ...],                    # preferred
    "dispreferred_audio_codes": [[...], ...]        # rejected (fixed generated output)
  }

`dispreferred_audio_codes` can also be provided as:
  - rejected_audio_codes
  - generated_audio_codes
  - reject_audio_codes
"""
import argparse
import json
import os
import shutil
from typing import Optional

import torch
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

from finetuning.sft_12hz_voice_design import VoiceDesignTTSDataset


class VoiceDesignDPODataset(VoiceDesignTTSDataset):
    def __init__(self, data_list, processor, config: Qwen3TTSConfig, lag_num: int = -1):
        super().__init__(data_list=data_list, processor=processor, config=config, lag_num=lag_num)
        self._reject_keys = [
            "dispreferred_audio_codes",
            "rejected_audio_codes",
            "generated_audio_codes",
            "reject_audio_codes",
        ]

    def _get_rejected_codes(self, item):
        for key in self._reject_keys:
            if key in item:
                return item[key]
        raise KeyError(
            "Missing rejected audio codes. Expected one of: "
            + ", ".join(self._reject_keys)
        )

    def __getitem__(self, idx):
        item = self.data_list[idx]

        text = item["text"]
        if "instruct" not in item:
            raise KeyError("Missing required field 'instruct' for voice design training.")
        instruct = item["instruct"]

        audio_codes = item["audio_codes"]
        rejected_audio_codes = self._get_rejected_codes(item)

        text = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text)

        instruct_ids: Optional[torch.Tensor]
        if instruct is None or instruct == "":
            instruct_ids = None
        else:
            instruct_text = self._build_instruct_text(instruct)
            instruct_ids = self._tokenize_texts(instruct_text)

        audio_codes = torch.tensor(audio_codes, dtype=torch.long)
        rejected_audio_codes = torch.tensor(rejected_audio_codes, dtype=torch.long)

        return {
            "text_ids": text_ids[:, :-5],              # 1, t
            "audio_codes": audio_codes,                # t, 16 (preferred)
            "rejected_audio_codes": rejected_audio_codes,  # t, 16 (rejected)
            "instruct_ids": instruct_ids,              # 1, t_ins or None
        }

    def _build_codec_batch(self, batch, audio_key: str):
        assert self.lag_num == -1

        item_length = [b["text_ids"].shape[1] + b[audio_key].shape[0] for b in batch]
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
            audio_codec_0 = data[audio_key][:, 0]
            audio_codecs = data[audio_key]

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

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
        }

    def collate_fn(self, batch):
        chosen = self._build_codec_batch(batch, "audio_codes")
        rejected = self._build_codec_batch(batch, "rejected_audio_codes")

        instruct_ids_list = [b["instruct_ids"] for b in batch]
        instruct_lengths = [ids.shape[1] if ids is not None else 0 for ids in instruct_ids_list]
        max_instruct_len = max(instruct_lengths) if instruct_lengths else 0

        bsz = len(batch)
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
            "chosen": chosen,
            "rejected": rejected,
            "instruct_ids": instruct_ids,
            "instruct_mask": instruct_mask,
        }


def _apply_instruct_prefix(
    model,
    input_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    codec_mask: torch.Tensor,
    codec_0_labels: torch.Tensor,
    codec_ids: torch.Tensor,
    instruct_ids: Optional[torch.Tensor],
    instruct_mask: Optional[torch.Tensor],
):
    if instruct_ids is None or instruct_ids.shape[1] == 0:
        return input_embeddings, attention_mask, codec_mask, codec_0_labels, codec_ids

    instruct_embeddings = model.talker.text_projection(
        model.talker.get_text_embeddings()(instruct_ids)
    )
    instruct_embeddings = instruct_embeddings * instruct_mask.unsqueeze(-1)

    input_embeddings = torch.cat([instruct_embeddings, input_embeddings], dim=1)
    prefix_len = instruct_ids.shape[1]
    attention_mask = torch.cat([instruct_mask.to(attention_mask.dtype), attention_mask], dim=1)
    codec_mask = torch.cat(
        [torch.zeros((codec_mask.shape[0], prefix_len), dtype=torch.bool, device=codec_mask.device), codec_mask],
        dim=1,
    )
    codec_0_labels = torch.cat(
        [
            torch.full((codec_0_labels.shape[0], prefix_len), -100, dtype=codec_0_labels.dtype, device=codec_0_labels.device),
            codec_0_labels,
        ],
        dim=1,
    )
    codec_ids = torch.cat(
        [
            torch.zeros((codec_ids.shape[0], prefix_len, codec_ids.shape[2]), dtype=codec_ids.dtype, device=codec_ids.device),
            codec_ids,
        ],
        dim=1,
    )

    return input_embeddings, attention_mask, codec_mask, codec_0_labels, codec_ids


def _sum_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    labels = labels.clone()
    mask = labels != -100
    labels[~mask] = 0
    token_logprobs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * mask
    return token_logprobs.sum(dim=1)


def _sum_logprobs_subtalker(
    model,
    talker_hidden_states: torch.Tensor,
    talker_codec_ids: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if talker_hidden_states.numel() == 0:
        return torch.zeros((batch_size,), device=talker_hidden_states.device)

    sub_talker_inputs_embeds = [talker_hidden_states.unsqueeze(1)]
    for i in range(model.talker.config.num_code_groups - 1):
        if i == 0:
            sub_talker_inputs_embeds.append(model.talker.get_input_embeddings()(talker_codec_ids[:, :1]))
        else:
            sub_talker_inputs_embeds.append(
                model.talker.code_predictor.get_input_embeddings()[i - 1](talker_codec_ids[:, i:i + 1])
            )
    sub_talker_inputs_embeds = torch.cat(sub_talker_inputs_embeds, dim=1)

    sub_outputs = model.talker.code_predictor.forward_finetune(
        inputs_embeds=sub_talker_inputs_embeds
    )
    sub_logits = sub_outputs.logits
    sub_labels = talker_codec_ids[:, 1:]

    sub_log_probs = F.log_softmax(sub_logits.float(), dim=-1)
    per_code_logp = sub_log_probs.gather(-1, sub_labels.unsqueeze(-1)).squeeze(-1)
    per_pos_logp = per_code_logp.sum(dim=1)

    agg = torch.zeros((batch_size,), device=per_pos_logp.device)
    agg.index_add_(0, batch_indices, per_pos_logp)
    return agg


def _compute_logps(model, config, batch_variant, instruct_ids, instruct_mask):
    input_ids = batch_variant["input_ids"]
    codec_ids = batch_variant["codec_ids"]
    text_embedding_mask = batch_variant["text_embedding_mask"]
    codec_embedding_mask = batch_variant["codec_embedding_mask"]
    attention_mask = batch_variant["attention_mask"]
    codec_0_labels = batch_variant["codec_0_labels"]
    codec_mask = batch_variant["codec_mask"]

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask

    pad_id = config.talker_config.codec_pad_id
    pad_embed = model.talker.model.codec_embedding.weight[pad_id]
    input_codec_embedding[:, 6, :] = pad_embed

    input_embeddings = input_text_embedding + input_codec_embedding
    for i in range(1, 16):
        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding

    input_embeddings, attention_mask, codec_mask, codec_0_labels, codec_ids = _apply_instruct_prefix(
        model,
        input_embeddings,
        attention_mask,
        codec_mask,
        codec_0_labels,
        codec_ids,
        instruct_ids,
        instruct_mask,
    )

    outputs = model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        output_hidden_states=True,
    )

    labels = codec_0_labels[:, 1:]
    main_logp = _sum_logprobs(outputs.logits, labels)

    hidden_states = outputs.hidden_states[0][-1]
    position_mask = codec_mask[:, 1:]
    talker_hidden_states = hidden_states[position_mask]
    talker_codec_ids = codec_ids[codec_mask]
    batch_indices = position_mask.nonzero(as_tuple=False)[:, 0]

    sub_logp = _sum_logprobs_subtalker(
        model,
        talker_hidden_states,
        talker_codec_ids,
        batch_indices,
        batch_size=codec_mask.shape[0],
    )

    return main_logp + sub_logp


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")

    model_path = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(model_path)

    ref_qwen3tts = Qwen3TTSModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    train_data = [json.loads(line) for line in open(args.train_jsonl).readlines()]
    dataset = VoiceDesignDPODataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    ref_model = ref_qwen3tts.model
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.to(accelerator.device)

    model.train()

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                instruct_ids = batch["instruct_ids"]
                instruct_mask = batch["instruct_mask"]

                policy_chosen_logp = _compute_logps(
                    model, config, batch["chosen"], instruct_ids, instruct_mask
                )
                policy_rejected_logp = _compute_logps(
                    model, config, batch["rejected"], instruct_ids, instruct_mask
                )

                with torch.no_grad():
                    ref_chosen_logp = _compute_logps(
                        ref_model, config, batch["chosen"], instruct_ids, instruct_mask
                    )
                    ref_rejected_logp = _compute_logps(
                        ref_model, config, batch["rejected"], instruct_ids, instruct_mask
                    )

                logits = (policy_chosen_logp - policy_rejected_logp) - (ref_chosen_logp - ref_rejected_logp)
                loss = -F.logsigmoid(args.beta * logits)
                if args.label_smoothing > 0:
                    loss = loss * (1.0 - args.label_smoothing) + (-F.logsigmoid(-args.beta * logits)) * args.label_smoothing
                loss = loss.mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                with torch.no_grad():
                    reward_chosen = args.beta * (policy_chosen_logp - ref_chosen_logp)
                    reward_rejected = args.beta * (policy_rejected_logp - ref_rejected_logp)
                    reward_margin = (reward_chosen - reward_rejected).mean().item()
                    accuracy = (reward_chosen > reward_rejected).float().mean().item()
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | "
                    f"RewardMargin: {reward_margin:.4f} | Acc: {accuracy:.4f}"
                )

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

# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm

from ...protocol import DataProto
from ...trainer import core_algos
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.compute_entropy_from_logits = torch.compile(VF.entropy_from_logits, dynamic=True)

    # def _forward_micro_batch(
    #     self, micro_batch: Dict[str, torch.Tensor], temperature: float
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Returns:
    #         entropy: # (bs, response_len)
    #         log_probs: # (bs, response_len)
    #     """
    #     input_ids = micro_batch["input_ids"]
    #     batch_size, seqlen = input_ids.shape
    #     attention_mask = micro_batch["attention_mask"]
    #     position_ids = micro_batch["position_ids"]
    #     responses = micro_batch["responses"]
    #     response_length = responses.size(-1)
    #     if position_ids.dim() == 3:  # qwen2vl mrope
    #         position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

    #     multi_modal_inputs = {}
    #     if "multi_modal_inputs" in micro_batch:
    #         for key in micro_batch["multi_modal_inputs"][0].keys():
    #             multi_modal_inputs[key] = torch.cat(
    #                 [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
    #             )

    #     with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    #         if self.config.padding_free:
    #             input_ids_rmpad, indices, *_ = unpad_input(
    #                 input_ids.unsqueeze(-1), attention_mask
    #             )  # input_ids_rmpad (total_nnz, ...)
    #             input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

    #             # unpad the position_ids to align the rotary
    #             if position_ids.dim() == 3:
    #                 position_ids_rmpad = (
    #                     index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
    #                     .transpose(0, 1)
    #                     .unsqueeze(1)
    #                 )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
    #             else:
    #                 position_ids_rmpad = index_first_axis(
    #                     rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
    #                 ).transpose(0, 1)

    #             # for compute the log_prob
    #             input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

    #             # pad and slice the inputs if sp > 1
    #             if self.config.ulysses_sequence_parallel_size > 1:
    #                 input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
    #                     input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_sequence_parallel_size
    #                 )
    #                 input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
    #                     input_ids_rmpad_rolled, None, self.config.ulysses_sequence_parallel_size
    #                 )

    #             input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

    #             # only pass input_ids and position_ids to enable flash_attn_varlen
    #             output = self.actor_module(
    #                 input_ids=input_ids_rmpad,
    #                 attention_mask=None,
    #                 position_ids=position_ids_rmpad,
    #                 **multi_modal_inputs,
    #                 use_cache=False,
    #             )  # prevent model thinks we are generating
    #             logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

    #             logits_rmpad.div_(temperature)

    #             # compute entropy
    #             entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

    #             # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
    #             log_probs = VF.logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

    #             # gather log_prob if sp > 1
    #             if self.config.ulysses_sequence_parallel_size > 1:
    #                 # gather and unpad for the ulysses sp
    #                 log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
    #                 entropy_rmpad = gather_outpus_and_unpad(
    #                     entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
    #                 )
    #             # pad back to (bsz, seqlen)
    #             full_entropy = pad_input(
    #                 hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
    #             )
    #             full_log_probs = pad_input(
    #                 hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
    #             )

    #             # only return response part
    #             entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
    #             log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
    #         else:
    #             output = self.actor_module(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 position_ids=position_ids,
    #                 **multi_modal_inputs,
    #                 use_cache=False,
    #             )
    #             logits: torch.Tensor = output.logits
    #             logits.div_(temperature)
    #             logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
    #             log_probs = VF.logprobs_from_logits(logits, responses)  # (bsz, response_length)
    #             entropy = VF.entropy_from_logits(logits)  # (bsz, response_length)

    #     return entropy, log_probs

    ###
    def _forward_micro_batch(
        self, micro_batch: Dict[str, torch.Tensor], temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes log probabilities and entropy for a micro-batch using OpenVLA model.

        Args:
            micro_batch: Dictionary containing input data:
                - "input_ids": [batch_size, prompt_length] (instruction tokens)
                - "attention_mask": [batch_size, prompt_length]
                - "pixel_values": [batch_size, 3, 224, 224] (image data)
                - "responses": [batch_size, 7] (action tokens)
            temperature: Scaling factor for logits (default 1.0)

        Returns:
            entropy: [batch_size, 7] (entropy of action token distribution)
            log_probs: [batch_size, 7] (log probabilities of selected action tokens)
        """
        input_ids = micro_batch["full_input_ids"]
        batch_size = input_ids.shape[0]
        attention_mask = micro_batch["extended_attention_mask"]
        responses = micro_batch["responses"]
        response_length = 7  # OpenVLA fixed length for 7-DoF actions

        # Prepare inputs for OpenVLA
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": micro_batch.get("pixel_values"),  # [batch_size, 3, 224, 224]
        }

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Forward pass with OpenVLA
            output = self.actor_module(**inputs, use_cache=False)
            # Extract logits for the last 7 tokens (action tokens)
            logits = output.logits[:, -response_length - 1 : -1, :]  # [batch_size, 7, 256]
            logits.div_(temperature)

            # Compute log probabilities and entropy
            log_probs = VF.logprobs_from_logits(logits=logits, labels=responses)  # [batch_size, 7]
            entropy = VF.entropy_from_logits(logits)  # [batch_size, 7]

        return entropy, log_probs
    ###

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        self.actor_optimizer.step()
        return grad_norm

    # @torch.no_grad()
    # def compute_log_prob(self, data: DataProto) -> torch.Tensor:
    #     """Compute the log probability of the responses given input_ids, attention_mask and position_ids

    #     Args:
    #         data (DataProto): a DataProto containing keys

    #             ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
    #             concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

    #             ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

    #             ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

    #             ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

    #     Returns:
    #         torch.Tensor: the log_prob tensor
    #     """
    #     self.actor_module.eval()

    #     temperature = data.meta_info["temperature"]
    #     select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
    #     if "multi_modal_inputs" in data.non_tensor_batch.keys():
    #         non_tensor_select_keys = ["multi_modal_inputs"]
    #     else:
    #         non_tensor_select_keys = []

    #     micro_batches = data.select(select_keys, non_tensor_select_keys).split(
    #         self.config.micro_batch_size_per_device_for_experience
    #     )
    #     log_probs_lst = []
    #     for micro_batch in tqdm(micro_batches, desc="Compute log probs", disable=(self.rank != 0)):
    #         model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
    #         _, log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
    #         log_probs_lst.append(log_probs)

    #     log_probs = torch.concat(log_probs_lst, dim=0)
    #     return log_probs

    ###
    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and pixel_values

        Args:
            data (DataProto): a DataProto containing keys
                "input_ids": [batch_size, prompt_length]
                "attention_mask": [batch_size, prompt_length]
                "pixel_values": [batch_size, 3, 224, 224]
                "responses": [batch_size, 7]

        Returns:
            torch.Tensor: the log_prob tensor [batch_size, 7]
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "full_input_ids", "extended_attention_mask", "pixel_values"]
        non_tensor_select_keys = []

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        for micro_batch in tqdm(micro_batches, desc="Compute log probs", disable=(self.rank != 0)):
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            _, log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs
    ###

    # def update_policy(self, data: DataProto) -> Dict[str, Any]:
    #     self.actor_module.train()

    #     temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
    #     select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
    #     if self.config.use_kl_loss:
    #         select_keys.append("ref_log_prob")

    #     if "multi_modal_inputs" in data.non_tensor_batch.keys():
    #         non_tensor_select_keys = ["multi_modal_inputs"]
    #     else:
    #         non_tensor_select_keys = []

    #     # Split to make minibatch iterator for updating the actor
    #     # See PPO paper for details. https://arxiv.org/abs/1707.06347
    #     mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

    #     metrics = defaultdict(list)
    #     n = len(mini_batches)
    #     for _ in range(self.config.ppo_epochs):
    #         for i, mini_batch in enumerate(mini_batches):
    #             gradient_accumulation = (
    #                 self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
    #             )
    #             micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

    #             self.actor_optimizer.zero_grad()
    #             for micro_batch in tqdm(micro_batches, desc=f"Update policy [{i + 1}/{n}]", disable=(self.rank != 0)):
    #                 model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
    #                 responses = model_inputs["responses"]
    #                 response_length = responses.size(1)
    #                 attention_mask = model_inputs["attention_mask"]
    #                 response_mask = attention_mask[:, -response_length:]
    #                 old_log_prob = model_inputs["old_log_probs"]
    #                 advantages = model_inputs["advantages"]

    #                 clip_ratio = self.config.clip_ratio
    #                 entropy_coeff = self.config.entropy_coeff

    #                 # all return: (bsz, response_length)
    #                 entropy, log_prob = self._forward_micro_batch(model_inputs, temperature=temperature)

    #                 pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
    #                     old_log_prob=old_log_prob,
    #                     log_prob=log_prob,
    #                     advantages=advantages,
    #                     eos_mask=response_mask,
    #                     cliprange=clip_ratio,
    #                 )
    #                 # compute entropy loss from entropy
    #                 entropy_loss = VF.masked_mean(entropy, response_mask)

    #                 # compute policy loss
    #                 policy_loss = pg_loss - entropy_loss * entropy_coeff

    #                 if self.config.use_kl_loss:
    #                     ref_log_prob = model_inputs["ref_log_prob"]
    #                     # compute kl loss
    #                     kld = core_algos.kl_penalty(
    #                         logprob=log_prob,
    #                         ref_logprob=ref_log_prob,
    #                         kl_penalty=self.config.kl_loss_type,
    #                     )
    #                     kl_loss = VF.masked_mean(kld, response_mask)
    #                     policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
    #                     metrics["actor/kl_loss"] = kl_loss.detach().item()
    #                     metrics["actor/kl_coef"] = self.config.kl_loss_coef

    #                 loss = policy_loss / gradient_accumulation
    #                 loss.backward()

    #                 batch_metrics = {
    #                     "actor/entropy_loss": entropy_loss.detach().item(),
    #                     "actor/pg_loss": pg_loss.detach().item(),
    #                     "actor/pg_clipfrac": pg_clipfrac.detach().item(),
    #                     "actor/ppo_kl": ppo_kl.detach().item(),
    #                 }
    #                 append_to_dict(metrics, batch_metrics)

    #             grad_norm = self._optimizer_step()
    #             append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

    #     self.actor_optimizer.zero_grad()
    #     return metrics

    ###
    # dp_actor.py
    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        """Update the policy using PPO algorithm for OpenVLA model.

        Args:
            data (DataProto): a DataProto containing keys
                "input_ids": [batch_size, prompt_length]
                "attention_mask": [batch_size, prompt_length]
                "pixel_values": [batch_size, 3, 224, 224]
                "responses": [batch_size, 7]
                "old_log_probs": [batch_size, 7]
                "advantages": [batch_size, 7]
                "ref_log_prob": [batch_size, 7] (if use_kl_loss)

        Returns:
            Dict: metrics dictionary containing loss, grad_norm, etc.
        """
        self.actor_module.train()

        torch.cuda.empty_cache()

        # 安全地获取 temperature，设置默认值为 1.0
        temperature = data.meta_info.get("temperature", 1.0)
        # 修改键名，与 _forward_micro_batch 保持一致
        select_keys = ["responses", "full_input_ids", "extended_attention_mask", "pixel_values", "advantages"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        non_tensor_select_keys = []

        # Split to make minibatch iterator for updating the actor
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        # 提前拆分 old_log probs 和 ref_log_prob，并确保设备一致
        old_log_prob_tensor = data.batch["old_log_probs"].batch["old_log_probs"]
        # 确保 old_log_prob_tensor 在 cuda:0 上
        old_log_prob_tensor = old_log_prob_tensor.to("cuda:0")
        print(f"Rank {self.rank} - old_log_prob_tensor device: {old_log_prob_tensor.device}, shape: {old_log_prob_tensor.shape}")
        # 拆分为 mini-batch
        old_log_prob_mini_batches = torch.split(old_log_prob_tensor, self.config.global_batch_size_per_device, dim=0)
        # 进一步拆分为 micro-batch
        old_log_prob_micro_batches = []
        for mini_batch in old_log_prob_mini_batches:
            micro_batches = torch.split(mini_batch, self.config.micro_batch_size_per_device_for_update, dim=0)
            old_log_prob_micro_batches.extend(micro_batches)

        if self.config.use_kl_loss:
            ref_log_prob_tensor = data.batch["ref_log_prob"].batch["ref_log_prob"]
            # 确保 ref_log_prob_tensor 在 cuda:0 上
            ref_log_prob_tensor = ref_log_prob_tensor.to("cuda:0")
            print(f"Rank {self.rank} - ref_log_prob_tensor device: {ref_log_prob_tensor.device}, shape: {ref_log_prob_tensor.shape}")
            # 拆分为 mini-batch
            ref_log_prob_mini_batches = torch.split(ref_log_prob_tensor, self.config.global_batch_size_per_device, dim=0)
            # 进一步拆分为 micro-batch
            ref_log_prob_micro_batches = []
            for mini_batch in ref_log_prob_mini_batches:
                micro_batches = torch.split(mini_batch, self.config.micro_batch_size_per_device_for_update, dim=0)
                ref_log_prob_micro_batches.extend(micro_batches)

        metrics = defaultdict(list)
        n = len(mini_batches)
        for _ in range(self.config.ppo_epochs):
            for i, mini_batch in enumerate(mini_batches):
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                self.actor_optimizer.zero_grad()
                for idx, micro_batch in enumerate(tqdm(micro_batches, desc=f"Update policy [{i + 1}/{n}]", disable=(self.rank != 0))):
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    # 打印 model_inputs 以调试
                    print(f"Rank {self.rank} - model_inputs keys: {list(model_inputs.keys())}")
                    for key, value in model_inputs.items():
                        if isinstance(value, torch.Tensor):
                            print(f"Rank {self.rank} - model_inputs[{key}]: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                        else:
                            print(f"Rank {self.rank} - model_inputs[{key}]: type={type(value)}")

                    responses = model_inputs["responses"]
                    response_length = 7  # OpenVLA fixed length
                    attention_mask = model_inputs["extended_attention_mask"]
                    response_mask = attention_mask[:, -response_length:]
                    # 使用拆分后的 old_log_prob
                    old_log_prob = old_log_prob_micro_batches[idx]
                    advantages = model_inputs["advantages"]

                    clip_ratio = self.config.clip_ratio
                    entropy_coeff = self.config.entropy_coeff

                    # Compute log probabilities and entropy
                    entropy, log_prob = self._forward_micro_batch(model_inputs, temperature=temperature)

                    # 打印 log_prob 和 old_log_prob 的设备和形状以调试
                    print(f"Rank {self.rank} - log_prob shape: {log_prob.shape}, dtype: {log_prob.dtype}, device: {log_prob.device}")
                    print(f"Rank {self.rank} - old_log_prob shape: {old_log_prob.shape}, dtype: {old_log_prob.dtype}, device: {old_log_prob.device}")

                    # Compute PPO policy loss
                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        eos_mask=response_mask,
                        cliprange=clip_ratio,
                    )
                    entropy_loss = VF.masked_mean(entropy, response_mask)

                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        # 使用拆分后的 ref_log_prob
                        ref_log_prob = ref_log_prob_micro_batches[idx]
                        print(f"Rank {self.rank} - ref_log_prob shape: {ref_log_prob.shape}, dtype: {ref_log_prob.dtype}, device: {ref_log_prob.device}")
                        kld = core_algos.kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = VF.masked_mean(kld, response_mask)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    loss = policy_loss / gradient_accumulation
                    loss.backward()

                    batch_metrics = {
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
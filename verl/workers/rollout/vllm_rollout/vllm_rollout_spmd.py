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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
"""

import sys
sys.path.append('/workspace/verl')

from contextlib import contextmanager
from typing import Any, List, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer
from vllm import LLM, RequestOutput, SamplingParams

from ....protocol import DataProto
from ....utils import torch_functional as VF
from ....utils.torch_dtypes import PrecisionType
from ..base import BaseRollout
from ..config import RolloutConfig

###
from transformers import AutoModelForVision2Seq, AutoProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer
###


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


# class vLLMRollout(BaseRollout):
#     def __init__(self, model_path: str, config: RolloutConfig, tokenizer: PreTrainedTokenizer):
#         """A vLLM rollout. It requires the module is supported by the vllm.

#         Args:
#             module: module here follows huggingface APIs
#             config: DictConfig
#             tokenizer: the task/model tokenizer
#         """
#         super().__init__()
#         self.config = config
#         self.pad_token_id = tokenizer.pad_token_id
#         if config.tensor_parallel_size > torch.distributed.get_world_size():
#             raise ValueError("Tensor parallelism size should be less than world size.")

#         if not config.enforce_eager and config.free_cache_engine:
#             raise ValueError("CUDA graph should be disabled when `free_cache_engine` is True.")

#         if config.max_num_batched_tokens < config.prompt_length + config.response_length:
#             raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

#         vllm_init_kwargs = {}
#         if config.limit_images > 0:
#             vllm_init_kwargs = {"limit_mm_per_prompt": {"image": config.limit_images}}

#         self.inference_engine = LLM(
#             model=model_path,
#             skip_tokenizer_init=False,
#             tensor_parallel_size=config.tensor_parallel_size,
#             dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
#             gpu_memory_utilization=config.gpu_memory_utilization,
#             enforce_eager=config.enforce_eager,
#             max_model_len=config.prompt_length + config.response_length,
#             max_num_batched_tokens=config.max_num_batched_tokens,
#             enable_sleep_mode=True,
#             distributed_executor_backend="external_launcher",
#             disable_custom_all_reduce=True,
#             disable_log_stats=config.disable_log_stats,
#             enable_chunked_prefill=config.enable_chunked_prefill,
#             **vllm_init_kwargs,
#         )

#         # Offload vllm model to reduce peak memory usage
#         self.inference_engine.sleep(level=1)

#         sampling_kwargs = {"max_tokens": config.response_length, "detokenize": False}
#         default_sampling_params = SamplingParams()
#         for key in config.to_dict().keys():
#             if hasattr(default_sampling_params, key):
#                 sampling_kwargs[key] = getattr(config, key)

#         print(f"Sampling params: {sampling_kwargs}.")
#         self.sampling_params = SamplingParams(**sampling_kwargs)

#     @contextmanager
#     def update_sampling_params(self, **kwargs):
#         # update sampling params
#         old_sampling_params_args = {}
#         if kwargs:
#             for key, value in kwargs.items():
#                 if hasattr(self.sampling_params, key):
#                     old_value = getattr(self.sampling_params, key)
#                     old_sampling_params_args[key] = old_value
#                     setattr(self.sampling_params, key, value)

#         yield
#         # roll back to previous sampling params
#         for key, value in old_sampling_params_args.items():
#             setattr(self.sampling_params, key, value)

#     @torch.no_grad()
#     def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
#         # left-padded attention_mask
#         input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
#         attention_mask: torch.Tensor = prompts.batch["attention_mask"]
#         position_ids: torch.Tensor = prompts.batch["position_ids"]
#         eos_token_id: int = prompts.meta_info["eos_token_id"]
#         batch_size = input_ids.size(0)

#         do_sample = prompts.meta_info.get("do_sample", True)
#         if not do_sample:
#             kwargs = {
#                 "n": 1,
#                 "temperature": 0.0,
#                 "top_p": 1.0,
#                 "top_k": -1,
#                 "min_p": 0.0,
#             }

#         non_tensor_batch = prompts.non_tensor_batch
#         if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
#             raise RuntimeError("vllm sharding manager is not work properly.")

#         if "multi_modal_data" in non_tensor_batch:
#             vllm_inputs = []
#             for raw_prompt_ids, multi_modal_data in zip(
#                 non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
#             ):
#                 vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
#         else:
#             vllm_inputs = [
#                 {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
#             ]

#         # users can customize different sampling_params at different run
#         with self.update_sampling_params(**kwargs):
#             completions: List[RequestOutput] = self.inference_engine.generate(
#                 prompts=vllm_inputs, sampling_params=self.sampling_params
#             )

#         response_ids = []
#         for completion in completions:
#             for output in completion.outputs:
#                 response_ids.append(output.token_ids)

#         response_ids = VF.pad_2d_list_to_length(
#             response_ids, self.pad_token_id, max_length=self.config.response_length
#         ).to(input_ids.device)

#         if self.config.n > 1 and do_sample:
#             batch_size = batch_size * self.config.n
#             input_ids = _repeat_interleave(input_ids, self.config.n)
#             attention_mask = _repeat_interleave(attention_mask, self.config.n)
#             position_ids = _repeat_interleave(position_ids, self.config.n)
#             if "multi_modal_inputs" in non_tensor_batch.keys():
#                 non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
#                     non_tensor_batch["multi_modal_inputs"], self.config.n
#                 )

#         sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
#         response_length = response_ids.size(1)
#         delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
#         delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
#         if position_ids.dim() == 3:  # qwen2vl mrope
#             delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

#         # prompt: left pad + response: right pad
#         # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
#         # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
#         response_position_ids = position_ids[..., -1:] + delta_position_id
#         position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
#         response_attention_mask = VF.get_eos_mask(
#             response_ids=response_ids, eos_token=eos_token_id, dtype=attention_mask.dtype
#         )
#         attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

#         # all the tp ranks should contain the same data here. data in all ranks are valid
#         batch = TensorDict(
#             {
#                 "prompts": input_ids,
#                 "responses": response_ids,
#                 "input_ids": sequence_ids,  # here input_ids become the whole sentences
#                 "attention_mask": attention_mask,
#                 "position_ids": position_ids,
#             },
#             batch_size=batch_size,
#         )
#         return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

# class vLLMRollout(BaseRollout):
#     def __init__(self, model_path: str, config: RolloutConfig, tokenizer):
#         """A rollout module for OpenVLA, generating action tokens for PPO training.

#         Args:
#             model_path (str): Path to the OpenVLA model (e.g., "openvla/openvla-7b").
#             config (RolloutConfig): Configuration for rollout settings.
#             tokenizer: The tokenizer associated with the model (replaced by processor.tokenizer).
#         """
#         super().__init__()
#         self.config = config

#         # Load OpenVLA model and processor
#         self.inference_engine = AutoModelForVision2Seq.from_pretrained(
#             model_path,
#             torch_dtype=torch.bfloat16,
#             device_map="cuda",
#             trust_remote_code=True,
#         )
#         self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
#         self.tokenizer = self.processor.tokenizer
#         self.pad_token_id = self.tokenizer.pad_token_id
#         self.action_tokenizer = ActionTokenizer(self.tokenizer)

#         # Validate response length for OpenVLA (7-DoF actions)
#         if self.config.response_length != 7:
#             raise ValueError("OpenVLA generates exactly 7 action tokens (7-DoF); set response_length to 7.")

#     def decode_actions(self, token_ids):
#         """Decode action tokens into continuous action values.

#         Args:
#             token_ids (torch.Tensor): Tensor of action token IDs, shape [batch_size, 7].

#         Returns:
#             torch.Tensor: Continuous action values, shape [batch_size, 7].
#         """
#         # Convert token IDs to continuous actions using ActionTokenizer
#         actions = self.action_tokenizer.decode_token_ids_to_actions(token_ids.cpu().numpy())
#         return torch.tensor(actions, dtype=torch.float32, device=token_ids.device)

#     @torch.no_grad()
#     def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
#         """Generate action token sequences using OpenVLA.

#         Args:
#             prompts (DataProto): Input prompts containing instructions and images.
#             kwargs: Additional keyword arguments (e.g., sampling parameters).

#         Returns:
#             DataProto: Output containing response_ids (action tokens) and continuous actions.
#         """
#         # Extract batch size
#         batch_size = len(prompts)
#         if batch_size == 0:
#             raise ValueError("Prompts batch size is 0; no data to process")

#         # Prepare inputs for OpenVLA
#         vllm_inputs = []
#         for p in prompts.non_tensor_batch:
#             if "language_instruction" not in p or "image" not in p:
#                 raise ValueError(f"Missing 'language_instruction' or 'image' in prompt: {p}")
#             instruction = p["language_instruction"]  # Text instruction
#             image = p["image"]  # PIL image or tensor
#             # Use OpenVLA processor to prepare inputs
#             inputs = self.processor(instruction, images=image, return_tensors="pt")
#             vllm_inputs.append(inputs)
        
#         print("vllm_inputs length:", len(vllm_inputs))
#         if vllm_inputs:
#             print("vllm_inputs[0] keys:", vllm_inputs[0].keys())
#             print("vllm_inputs[0]['input_ids'] shape:", vllm_inputs[0]["input_ids"].shape)
#             print("vllm_inputs[0]['pixel_values'] shape:", vllm_inputs[0]["pixel_values"].shape)
        
#         if not vllm_inputs:
#             raise ValueError("vllm_inputs is empty; no valid prompts processed")

#         # Combine batch inputs
#         inputs = {
#             "input_ids": torch.cat([x["input_ids"] for x in vllm_inputs], dim=0).to("cuda"),
#             "attention_mask": torch.cat([x["attention_mask"] for x in vllm_inputs], dim=0).to("cuda"),
#             "pixel_values": torch.cat([x["pixel_values"] for x in vllm_inputs], dim=0).to("cuda"),
#         }

#         # OpenVLA inference
#         outputs = self.inference_engine(**inputs)
#         response_ids = outputs.logits[:, -7:, :].argmax(dim=-1)  # [batch_size, 7]

#         # Decode actions for reward computation
#         actions = self.decode_actions(response_ids)  # [batch_size, 7]

#         # Handle repeat for PPO rollout (if n > 1)
#         if self.config.n > 1 and kwargs.get("do_sample", True):
#             batch_size = batch_size * self.config.n
#             input_ids = _repeat_interleave(inputs["input_ids"], self.config.n)
#             attention_mask = _repeat_interleave(inputs["attention_mask"], self.config.n)
#             response_ids = _repeat_interleave(response_ids, self.config.n)
#             actions = _repeat_interleave(actions, self.config.n)

#         # Construct DataProto for PPO
#         sequence_ids = torch.cat([inputs["input_ids"], response_ids], dim=-1)
#         response_length = response_ids.size(1)
#         delta_position_id = torch.arange(1, response_length + 1, device=inputs["input_ids"].device)
#         delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
#         response_position_ids = inputs["input_ids"][..., -1:] + delta_position_id
#         position_ids = torch.cat([inputs["input_ids"], response_position_ids], dim=-1)
#         response_attention_mask = torch.ones_like(response_ids, dtype=attention_mask.dtype)
#         attention_mask = torch.cat((inputs["attention_mask"], response_attention_mask), dim=-1)

#         batch = TensorDict(
#             {
#                 "prompts": inputs["input_ids"],
#                 "responses": response_ids,
#                 "actions": actions,  # Continuous actions for reward computation
#                 "input_ids": sequence_ids,
#                 "attention_mask": attention_mask,
#                 "position_ids": position_ids,
#             },
#             batch_size=batch_size,
#         )
#         return DataProto(batch=batch, non_tensor_batch=prompts.non_tensor_batch)

class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer):
        """A rollout module for OpenVLA, generating action tokens for PPO training.

        Args:
            model_path (str): Path to the OpenVLA model (e.g., "openvla/openvla-7b").
            config (RolloutConfig): Configuration for rollout settings.
            tokenizer: The tokenizer associated with the model (replaced by processor.tokenizer).
        """
        super().__init__()
        self.config = config

        # Load OpenVLA model and processor
        self.inference_engine = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.action_tokenizer = ActionTokenizer(self.tokenizer)

        # Validate response length for OpenVLA (7-DoF actions)
        if self.config.response_length != 7:
            raise ValueError("OpenVLA generates exactly 7 action tokens (7-DoF); set response_length to 7.")

    def decode_actions(self, token_ids):
        """Decode action tokens into continuous action values.

        Args:
            token_ids (torch.Tensor): Tensor of action token IDs, shape [batch_size, 7].

        Returns:
            torch.Tensor: Continuous action values, shape [batch_size, 7].
        """
        actions = self.action_tokenizer.decode_token_ids_to_actions(token_ids.cpu().numpy())
        return torch.tensor(actions, dtype=torch.float32, device=token_ids.device)

    # @torch.no_grad()
    # def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    #     """Generate action token sequences using OpenVLA.

    #     Args:
    #         prompts (DataProto): Input prompts containing instructions and images.
    #         kwargs: Additional keyword arguments (e.g., sampling parameters).

    #     Returns:
    #         DataProto: Output containing response_ids (action tokens) and continuous actions.
    #     """
    #     # Extract batch size
    #     batch_size = len(prompts)
    #     if batch_size == 0:
    #         raise ValueError("Prompts batch size is 0; no data to process")

    #     # Prepare inputs for OpenVLA
    #     vllm_inputs = []
    #     for p in prompts.non_tensor_batch:
    #         if "language_instruction" not in p or "image" not in p:
    #             raise ValueError(f"Missing 'language_instruction' or 'image' in prompt: {p}")
    #         instruction = p["language_instruction"]  # Text instruction
    #         image = p["image"]  # PIL image or tensor
    #         # Use OpenVLA processor to prepare inputs
    #         inputs = self.processor(instruction, images=image, return_tensors="pt")
    #         vllm_inputs.append(inputs)

    #     print("vllm_inputs length:", len(vllm_inputs))
    #     if vllm_inputs:
    #         print("vllm_inputs[0] keys:", vllm_inputs[0].keys())
    #         print("vllm_inputs[0]['input_ids'] shape:", vllm_inputs[0]["input_ids"].shape)
    #         print("vllm_inputs[0]['pixel_values'] shape:", vllm_inputs[0]["pixel_values"].shape)

    #     if not vllm_inputs:
    #         raise ValueError("vllm_inputs is empty; no valid prompts processed")

    #     # Combine batch inputs
    #     inputs = {
    #         "input_ids": torch.cat([x["input_ids"] for x in vllm_inputs], dim=0).to("cuda"),
    #         "attention_mask": torch.cat([x["attention_mask"] for x in vllm_inputs], dim=0).to("cuda"),
    #         "pixel_values": torch.cat([x["pixel_values"] for x in vllm_inputs], dim=0).to("cuda"),
    #     }

    #     # OpenVLA inference
    #     outputs = self.inference_engine(**inputs)
    #     response_ids = outputs.logits[:, -7:, :].argmax(dim=-1)  # [batch_size, 7]

    #     # Decode actions for reward computation
    #     actions = self.decode_actions(response_ids)  # [batch_size, 7]

    #     # Handle repeat for PPO rollout (if n > 1)
    #     if self.config.n > 1 and kwargs.get("do_sample", True):
    #         batch_size = batch_size * self.config.n
    #         input_ids = _repeat_interleave(inputs["input_ids"], self.config.n)
    #         attention_mask = _repeat_interleave(inputs["attention_mask"], self.config.n)
    #         response_ids = _repeat_interleave(response_ids, self.config.n)
    #         actions = _repeat_interleave(actions, self.config.n)

    #     # Construct DataProto for PPO
    #     sequence_ids = torch.cat([inputs["input_ids"], response_ids], dim=-1)
    #     attention_mask = torch.cat([inputs["attention_mask"], torch.ones_like(response_ids)], dim=-1)

    #     batch = TensorDict(
    #         {
    #             "prompts": inputs["input_ids"],
    #             "responses": response_ids,
    #             "actions": actions,  # Continuous actions for reward computation
    #             "input_ids": sequence_ids,
    #             "attention_mask": attention_mask,
    #         },
    #         batch_size=batch_size,
    #     )
    #     return DataProto(batch=batch, non_tensor_batch=prompts.non_tensor_batch)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        batch_size = len(prompts)
        if batch_size == 0:
            raise ValueError("Prompts batch size is 0; no data to process")

        # 直接使用 batch 中的预处理数据
        inputs = {
            "input_ids": prompts.batch["input_ids"].to("cuda", dtype=torch.long),  # 保持为 torch.long
            "attention_mask": prompts.batch["attention_mask"].to("cuda", dtype=torch.long),  # 保持为 torch.long
            "pixel_values": prompts.batch["pixel_values"].to("cuda", dtype=torch.bfloat16),
        }

        print("inputs['input_ids'] shape:", inputs["input_ids"].shape)
        print("inputs['input_ids'] dtype:", inputs["input_ids"].dtype)
        print("inputs['pixel_values'] shape:", inputs["pixel_values"].shape)
        print("inputs['pixel_values'] dtype:", inputs["pixel_values"].dtype)

        with torch.no_grad():
            outputs = self.inference_engine(**inputs)
        response_ids = outputs.logits[:, -7:, :].argmax(dim=-1)
        actions = self.decode_actions(response_ids)

        print("response_ids shape:", response_ids.shape)
        print("response_ids sample:", response_ids[0])
        print("actions shape:", actions.shape)
        print("actions sample:", actions[0])

        sequence_ids = torch.cat([inputs["input_ids"], response_ids], dim=-1)
        attention_mask = torch.cat([inputs["attention_mask"], torch.ones_like(response_ids)], dim=-1)
        batch = TensorDict(
            {
                "prompts": inputs["input_ids"],
                "responses": response_ids,
                "predicted_actions": actions,  # 重命名为 predicted_actions
                "full_input_ids": sequence_ids,  # 重命名为 full_input_ids
                "extended_attention_mask": attention_mask,  # 重命名为 extended_attention_mask
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=prompts.non_tensor_batch)
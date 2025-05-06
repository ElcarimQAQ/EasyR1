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

import json

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import CustomRewardManager
from .config import PPOConfig
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        print(json.dumps(config.to_dict(), indent=2))###
        # 加载 OpenVLA 的 processor 和 tokenizer
        from transformers import AutoProcessor
        # processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(
            # ppo_config.worker.actor.model.model_path,  # /workspace/models/openvla-7b
            config.worker.actor.model.model_path,
            trust_remote_code=True,
            local_files_only=True,  # 强制本地加载
        )
        tokenizer = processor.tokenizer    
        # 实例化 tokenizer 和 processor        
         # 已经通过 processor 获取，无需额外加载
        # 如果需要单独加载 tokenizer，可以使用：
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
###

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=False,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
            Role.RefPolicy: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        reward_fn = CustomRewardManager(
            tokenizer=tokenizer, num_examine=1, compute_score=config.worker.reward.compute_score
        )
        val_reward_fn = CustomRewardManager(
            tokenizer=tokenizer, num_examine=1, compute_score=config.worker.reward.compute_score
        )


        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            # train_dataloader=train_dataloader,
            # val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "PYTHONUNBUFFERED": "1",
            }
        }
        # ray.init(runtime_env=runtime_env)
        ray.init(runtime_env=runtime_env, local_mode=True)
        print("Ray initialized successfully.")
        print("Starting main_task...")
        print("Ray cluster resources:", ray.cluster_resources())
        print("Ray available resources:", ray.available_resources())

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))
    print("main_task completed.")


if __name__ == "__main__":
    main()
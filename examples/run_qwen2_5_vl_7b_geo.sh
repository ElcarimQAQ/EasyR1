set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export NCCL_DEBUG=WARN
# export WANDB_API_KEY='e2e0261992fab6ce73072eaac4424f5a7e29a8f1'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export WANDB_API_KEY='e2e0261992fab6ce73072eaac4424f5a7e29a8f1'
MODEL_PATH=/data/models/Qwen2.5-VL-7B-Instruct   # replace it with your local file path
export CUDA_VISIBLE_DEVICES=0,1,2,3
SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=examples/grpo_example.yaml \
    data.train_files=/data/datasets/geometry3k@train \
    data.val_files=/data/datasets/geometry3k@test \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_7b_geo \
    trainer.n_gpus_per_node=4

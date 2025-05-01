set -x

# export VLLM_ATTENTION_BACKEND=XFORMERS
# export VLLM_USE_V1=0
export PYTHONPATH=/workspace/verl
export NCCL_DEBUG=WARN
export MASTER_ADDR=172.18.0.2
export MASTER_PORT=63799
# export WANDB_API_KEY='e2e0261992fab6ce73072eaac4424f5a7e29a8f1'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false  # RLDS 要求关闭并行
# export TOKENIZERS_PARALLELISM=true
export WANDB_API_KEY='e2e0261992fab6ce73072eaac4424f5a7e29a8f1'
MODEL_PATH=/workspace/models/openvla-7b   # replace it with your local file path
export CUDA_VISIBLE_DEVICES=1,2
SYSTEM_PROMPT="""Act as a robot to perform the given task."""


ray job submit --address="http://127.0.0.1:8265" \
    --no-wait \
    -- \
    python3 -m verl.trainer.main \
    config=examples/grpo_example_vla.yaml \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=openvla_7b_robot \
    trainer.n_gpus_per_node=2

    # data.train_files=/workspace/bridge_dataset/1.0.0/bridge_dataset-train.tfrecord-* \
    # data.val_files=/workspace/bridge_dataset/1.0.0/bridge_dataset-val.tfrecord-* \

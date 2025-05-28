python -m llm_sft.answer_eval \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --model_type remote \
    --platform VLLM \
    --input_path /the-path-to-your-prepared-data \
    --image_dir /the-path-to-your-image-dir\
   
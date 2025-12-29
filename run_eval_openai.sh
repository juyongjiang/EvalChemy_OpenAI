python -m eval.eval \
    --model openai_api \
    --tasks HumanEval,HumanEvalPlus,MBPP,MBPPPlus,CodeElo,LiveCodeBenchv5_official,BigCodeBench \
    --model_args "model=GPT-5,base_url=https://api.openai.com/v1,num_concurrent=64" \
    --batch_size 1 \
    --max_tokens 16384 \
    --apply_chat_template True \
    --output_path logs
## Installation
```bash
git clone git@github.com:mlfoundations/evalchemy.git

conda create --name evalchemy python=3.10
conda activate evalchemy
python -m pip install --upgrade pip
pip install -e .
```

## Support OpenAI-like Models Evaluation

1. Move `openai_api.py` file to `eval` folder;
2. Write `from eval.openai_api import OpenAIAPIModel` into the beginning of `eval/eval.py` file to register openai_api model;
3. Run the following commands to start evaluation:

```bash
python -m eval.eval \
    --model openai_api \
    --tasks HumanEval,HumanEvalPlus,MBPP,MBPPPlus,CodeElo,LiveCodeBenchv5_official,BigCodeBench \
    --model_args "model=GPT-5,base_url=https://api.openai.com/v1,num_concurrent=64" \
    --batch_size 1 \
    --max_tokens 16384 \
    --apply_chat_template True \
    --output_path logs
```

## Evaluation Results

| Institution | Model | HE | HE+ | MBPP | MBPP+ | BBC | LCB | CF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **OpenAI** | GPT-5 |  |  |  |  |  |  |  |
| **OpenAI** | GPT4.1 |  |  |  |  |  |  |  |
| **Anthropic** | Claude-Sonnet-4.5 |  |  |  |  |  |  |  |
| **Google** | Gemini-2.5-Pro |  |  |  |  |  |  |  |
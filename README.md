<div align="center">

  <h1>
    <img src="assets/logo.svg" height="40px" style="vertical-align: middle;">
    SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning
  </h1>

  <p><em>A novel framework that enhances the reasoning capabilities of multimodal large language models</em></p>

  <p>If you find this project useful, please give us a star 🌟.</p>

 <p>
  <a href="https://arxiv.org/abs/xxxx.xxxxx">
    <img src="https://img.shields.io/badge/Arxiv-Paper-red?logo=arxiv">
  </a>
  <a href="https://huggingface.co/datasets/SRPOMLLMs/srpo-sft-data">
    <img src="https://img.shields.io/badge/Hugging%20Face-Models-blue?logo=huggingface">
  </a>
  <a href="https://huggingface.co/datasets/SRPOMLLMs/srpo-sft-data">
    <img src="https://img.shields.io/badge/Hugging%20Face-Dataset-yellow?logo=huggingface">
  </a>
  <a href="https://srpo.pages.dev">
    <img src="https://img.shields.io/badge/Project-Page-orange?logo=googlechrome&logoColor=FFCD00">
  </a>
</p>


  <p>
    <a href="https://scholar.google.com/citations?hl=en&user=EVj1cNoAAAAJ">Zhongwei Wan</a><sup>2†*✉️</sup>,
    <a href="https://www.linkedin.com/in/zhihao-dou-760276261/">Zhihao Dou</a><sup>3†</sup>,
    <a href="https://scholar.google.com/citations?user=HED_458AAAAJ&hl=zh-CN">Che Liu</a><sup>4</sup>,
    Yu Zhang<sup>11</sup>,
    <a href="https://dongfeicui.github.io">Dongfei Cui</a><sup>5</sup>,
    <a href="https://github.com/AlbertZhaoCA">Qinjian Zhao</a><sup>6</sup>,
    <a href="https://nastymarcus.github.io">Hui Shen</a><sup>7</sup>,
    <a href="https://menik1126.github.io">Jing Xiong</a><sup>10</sup>,
    <a href="https://synbol.github.io">Yi Xin</a><sup>12</sup>,
    <a href="https://yifanjiang-921.github.io">Yifan Jiang<sup>8</sup>,
    <a href="https://scholar.google.com/citations?user=gjmfLroAAAAJ&hl=zh-CN">Chaofan Tao</a><sup>10</sup>,
    <a href="https://github.com/codepassionor">Yangfan He</a><sup>9</sup>,
    <a href="https://mi-zhang.github.io">Mi Zhang</a><sup>2</sup>,
    <a href="https://shenyann.github.io">Shen Yan</a><sup>1✉️</sup>
  </p>

  <p>
    <sup>1</sup>
    <img src="assets/bytedance-seed.svg" height="25px" style="vertical-align: middle; margin-right: 24px;">
    <sup>2</sup>
    <img src="assets/osu2.png" height="25px" style="vertical-align: middle;">
  </p>

  <p>
    <sup>3</sup>Case Western Reserve University,
    <sup>4</sup>Imperial College London,
    <sup>5</sup>Duke University,
    <sup>6</sup>Kean University,
    <sup>7</sup>University of Michigan,
    <sup>8</sup>University of Southern California,
    <sup>9</sup>University of Minnesota,
    <sup>10</sup>The University of Hong Kong,
    <sup>11</sup>Tongji University,
    <sup>12</sup>Nanjing University
  </p>

  <p><sup>*</sup>Project Leader (work completed during internship at Bytedance), <sup>†</sup>Equal Contribution, <sup>✉️</sup>Corresponding Author, </p>

<div>Corresponding to <sup>2</sup><a href="mailto:wan.512@osu.edu">wan.512@osu.edu</a>, <sup>1</sup><a href="mailto:sheny@bytedance.com">sheny@bytedance.com</a>
</div>
</div>

## 🔥 Quick Start

## Self-Reflection SFT Data Curation

```bash
# Clone the repository
git clone https://github.com/SUSTechBruce/SRPO_MLLMs
cd SRPO_MLLMs

# Install dependencies
pip install -r requirements.txt
```

### 1. Data Preparation
- Download data from [Mulberry-SFT](https://huggingface.co/datasets/HuanjinYao/Mulberry-SFT) and [LLaVA-CoT-100k](https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k), or prepare your own dataset in a similar format.
- Place your input data (e.g., `input.jsonl`) in a designated data directory (such as `data/`).

**Example (LLaVA-CoT-100k format):**
```json
{
  "query": "How many Mexican municipal leaders were killed in the previous year? Answer the question using a single word or phrase.",
  "image": "chartqa/train/png/two_col_100466.png",
  "answer": "21",
  "content": "<SUMMARY> I will examine the image to determine the number of Mexican municipal leaders killed in the previous year by analyzing the data presented in the bar chart. </SUMMARY>\n\n<CAPTION> The image displays a bar chart illustrating the number of Mexican municipal leaders killed each year from 2005 to 2018. Each bar represents the total number of victims for a specific year. </CAPTION>\n\n<REASONING> I will look at the bar corresponding to the year 2017 to find the number of Mexican municipal leaders killed in the previous year. The chart indicates that in 2017, there were 21 victims, as shown by the height of the bar labeled for that year. </REASONING>\n\n<CONCLUSION> 21 </CONCLUSION>"
}
```
- Your data must include at least the fields: `query`, `answer`, and `image`. The `content` field (as in Mulberry-SFT and LLaVA-CoT-100k) is used for image description extraction (optional).
- Place images in a folder (e.g., `images/`).
- For multimodal tasks, ensure the `image` field in your input file contains the correct relative path or URL to the image.

### 2. Data Construction

#### Answer Evaluation
```bash
python -m llm_sft.answer_eval \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --model_type remote \
    --platform VLLM \
    --input_path /path/to/your/data.jsonl \
    --image_dir /path/to/your/images
```
> **Note:**
> This command runs the LLM to answer the queries in your prepared data.

#### Reflection Evaluation
```bash
python -m llm_sft.reflection_eval \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --model_type remote \
    --platform VLLM \
    --input_path /path/to/your/data.jsonl \
    --image_dir /path/to/your/images \
    --output_path /path/to/save/reflections.jsonl
```
> **Note:**
> - This command lets the advanced MLLM generate reflections for each sample.
> - If you use `openai` or `azure` as the platform, images will be automatically encoded as base64 and sent to the API by default.
> - For large images or to avoid base64 encoding, you can upload your images to a public server or image hosting service, then set the `--image_url` argument to the accessible URL prefix.
> - Alternatively, you can implement your own upload logic in `utils/upload_utils.py` and use the `--upload_image` flag to enable custom image uploading.

#### Image Description Extraction
```bash
python -m llm_sft.image_description \
    --input_path /path/to/your/data.jsonl \
    --source cot100k \
    --output_path /path/to/save/image_descriptions.jsonl
```
> **Note:**
> - Run this only if you want to use unimodal models (e.g., o3-mini) for reflection, or need to extract image descriptions for other purposes.
> - You can extract image descriptions from [Mulberry-SFT](https://huggingface.co/datasets/HuanjinYao/Mulberry-SFT) and [LLaVA-CoT-100k](https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k) using our predefined patterns, or from your own dataset with a custom pattern.

### 3. Output
- Results and checkpoints are saved as JSONL files in the specified locations.
- Each result contains the question, image, model answer, standard answer, and reasoning chain.

### 4. Workflow
You can also run the shell scripts provided in the `/scripts` directory (such as `eval_answer.sh`, `eval_reflection.sh`, `eval_extract_description.sh`) for one-click batch evaluation and image description extraction.

---

### 5. Reproducibility
You can use the SFT data we provide in our [Hugging Face dataset](https://huggingface.co/SRPOMLLMs), or prepare your own using the methods described above.


## Dataset
> **Self-reflection SFT** dataset (for Self-reflection Supervised Fine-Tuning):  
> [srpo-sft-data on Hugging Face Datasets](https://huggingface.co/datasets/SRPOMLLMs/srpo-sft-data)

> **Self-reflection RL** dataset (for Self-reflection Reinforcement Learning):  
> [SRPO_RL_datasets on Hugging Face Datasets](https://huggingface.co/datasets/bruce360568/SRPO_RL_datasets)


## Self-Reflection Cold Start 
After you preprocess self-reflection sft data, please install LLaMA-Factory for Self-Reflection SFT:

```bash
cd SRPO_MLLMs/srpo_sft/LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```
Then run:

```bash
llamafactory-cli train examples/train_full/qwen2_5vl_full_sft.yaml ## for 7B
llamafactory-cli train examples/train_full/qwen2_5vl_full_sft_32b.yaml ## for 32B
```

## Self-Reflection RL Training

After the self-reflection SFT stage, we obtain updated model weights. Based on these weights, we then conduct self-reflection RL training. We provide implementations in both the  `OpenRLHF` and `Verl` frameworks (with the results reported in the main paper derived from the OpenRLHF version).

### OpenRLHF Version
Install the `OpenRLHF` Version
```bash
cd SRPO_MLLMs/spro_rl_train/openrlhf_srpo
pip install -e .[vllm]
pip install flash_attn --no-build-isolation
```

Start to train:
```bash
sh examples/scripts/run_7b_sft_srpo_filter_data.sh  # for 7B
sh examples/scripts/run_32b_sft_srpo.sh  # for 32B
```



### Verl Version

Install the `Verl` Version and then transfer the data format following [Link](https://huggingface.co/datasets/hiyouga/geometry3k)
```bash
cd SRPO_MLLMs/spro_rl_train/verl_srpo
pip install -e .
```

Start to train:
```bash
sh examples/qwen2_5_vl_7b_srpo.sh  # for 7B
sh examples/qwen2_5_vl_32b_srpo.sh  # for 32B
```

### Easy Step to Evaluation

```bash
cd SRPO_MLLMs/spro_rl_train/openrlhf_srpo/eval/mathverse
python evaluate_mathverse.py
python mathverse/extract_calculate.py --output_file xxx.json
```
For the results reported in the paper, we adopt the benchmark test data from [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## Acknowledgements

We acknowledge the outstanding open-source contributions from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [Verl](https://github.com/volcengine/verl), [EasyR1](https://github.com/hiyouga/EasyR1) for their open-source techniques and base models, which have enabled us to further our exploration.

## 📄 Citation
If you use SRPO or this codebase, please cite our paper:

```bibtex
@article{wan2025srpo,
  title={Srpo: Enhancing multimodal llm reasoning via reflection-aware reinforcement learning},
  author={Wan, Zhongwei and Dou, Zhihao and Liu, Che and Zhang, Yu and Cui, Dongfei and Zhao, Qinjian and Shen, Hui and Xiong, Jing and Xin, Yi and Jiang, Yifan and others},
  journal={arXiv preprint arXiv:2506.01713},
  year={2025}
}
```

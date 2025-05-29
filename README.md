<div align="center">

  <h1>
    <img src="assets/logo.svg" height="40px" style="vertical-align: middle;">
    SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning
  </h1>

  <p><em>A novel framework that enhances the reasoning capabilities of multimodal large language models</em></p>

  <p>If you find this project useful, please give us a star 🌟.</p>

  <p>
    <a href="https://arxiv.org/abs/xxxx.xxxxx">
      <img src="https://img.shields.io/badge/Paper-Arxiv-red?logo=arxiv">
    </a>
    <a href="https://huggingface.co/SRPOMLLMs">
      <img src="https://img.shields.io/badge/Hugging%20Face-Models-blue?logo=huggingface">
    </a>
    <a href="https://huggingface.co/SRPOMLLMs/SRPO-Dataset">
      <img src="https://img.shields.io/badge/Dataset-Huggingface-yellow?logo=huggingface">
    </a>
  </p>

  <p>
    <a href="https://scholar.google.com/citations?hl=en&user=EVj1cNoAAAAJ">Zhongwei Wan</a><sup>2†*✉️</sup>,
    Zhihao Dou<sup>3†</sup>,
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

## TODO: Self-Reflection Cold Start 

## TODO: Self-Reflection RL Training

## 📄 Citation
If you use SRPO or this codebase, please cite our paper:

```bibtex
placeholder
```

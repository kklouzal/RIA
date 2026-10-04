---
pipeline_tag: image-text-to-text
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash
license: mit
library_name: Model Optimizer
tags:
- nvidia
- ModelOpt
- DeepSeekV4
- quantized
- FP4
- fp4
---

# Model Overview

## Description:
The NVIDIA DeepSeek-V4.1-Flash NVFP4 model is the quantized version of DeepSeek AI's DeepSeek-V4.1-Flash model, which is an auto-regressive language model that uses an optimized transformer architecture. DeepSeek-V4.1-Flash is a natively multimodal Mixture-of-Experts (MoE) model with a Causal Encoder-Decoder (CED) architecture, Compressed Sparse Attention 2 (CSA2), and support for contexts of up to one million tokens. For more information, please check [here](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash). The NVIDIA DeepSeek-V4.1-Flash NVFP4 model is quantized with [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).

This model is ready for commercial or non-commercial use.  <br>

## Third-Party Community Consideration
This model is not owned or developed by NVIDIA. This model has been developed and built to a third-party's requirements for this application and use case; see link to Non-NVIDIA [(DeepSeek-V4.1-Flash) Model Card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) from DeepSeek AI.

### License/Terms of Use:
[MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)

### Deployment Geography:
Global <br>

### Use Case:
Developers looking to take off-the-shelf, pre-quantized models for deployment in AI Agent systems, chatbots, RAG systems, and other AI-powered applications. <br>

### Release Date:
Hugging Face 09/16/2026 via [https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4) <br>

## References
NVIDIA Model Optimizer: https://github.com/NVIDIA/Model-Optimizer

## Model Architecture:
**Architecture Type:** Transformers  <br>
**Network Architecture:** DeepSeek-V4.1-Flash (`DeepseekV41ForCausalLM`) — Causal Encoder-Decoder Mixture-of-Experts with Compressed Sparse Attention 2 <br>
**Number of Model Parameters:** 552B backbone in total, 8B activated during prefill and 16B during decode, plus 196B Engram conditional memory <br>
**This model was developed based on [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)** <br>

## Input:
**Input Type(s):** Text, Image <br>
**Input Format(s):** String, Red, Green, Blue (RGB) <br>
**Input Parameters:** One-Dimensional (1D), Two-Dimensional (2D) <br>
**Other Properties Related to Input:** Context length up to 1M <br>

## Output:
**Output Type(s):** Text <br>
**Output Format:** String <br>
**Output Parameters:** One-Dimensional (1D): Sequences<br>
**Other Properties Related to Output:** None <br>

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br>

## Software Integration:
**Supported Runtime Engine(s):** <br>
* **vLLM** <br>
* **SGLang** <br>

Both runtimes have been tested with this NVFP4 checkpoint on GB300. The benchmark results below were obtained with vLLM; SGLang was validated through loading, generation, reasoning/tool-call parsing, and image-input smoke tests.

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Blackwell <br>

**Preferred Operating System(s):** <br>
* Linux <br>

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):
The model is quantized with nvidia-modelopt **v0.47.0rc0** and V4.1-specific compatibility changes. Source revision: `dba1be0a40aa45a94ad051997016db3960a90277`. <br>

## Training and Evaluation Datasets:

## Calibration Dataset:
** Link: [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail), [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) <br>
** Data Collection Method by dataset: Automated. <br>
** Labeling Method by dataset: Automated. <br>
** Properties: The cnn_dailymail dataset is an English-language dataset containing just over 300k unique news articles as written by journalists at CNN and the Daily Mail. The Nemotron-Post-Training-Dataset-v2 is a post-training dataset curated by NVIDIA containing multi-turn conversations across diverse topics. <br>

Calibration used 512 samples per dataset (1,024 total), sequence length 512, batch size 4, and selection seed 0. The Nemotron subsets were `stem`, `chat`, `math`, and `code`.

## Training Dataset:
** Data Modality: Undisclosed <br>
** Data Collection Method by dataset: Undisclosed <br>
** Labeling Method by dataset: Undisclosed <br>
** Properties: Undisclosed

## Evaluation Dataset:
* Datasets: GPQA Diamond, AA-LCR, SciCode, IFBench, MMMU-Pro, Terminal-Bench 2.1 <br>
** Data Collection Method by dataset: Hybrid: Automated, Manually-Collected <br>
** Labeling Method by dataset: Hybrid: Manually-Labeled, Automated <br>
** Properties: These benchmarks cover scientific reasoning (GPQA Diamond), long-context recall (AA-LCR), scientific coding (SciCode), instruction following (IFBench), visual understanding (MMMU-Pro), and terminal-based agent tasks (Terminal-Bench 2.1). <br>

## Inference:
**Acceleration Engines:** **vLLM and SGLang** <br>
**Test Hardware:** **NVIDIA GB300 (Blackwell)**<br>

## Post Training Quantization
This model converts the ordinary routed MoE experts from source MXFP4 to NVFP4 weights and activations (W4A4), with group size 16. The converted projections are `w1`, `w2`, and `w3` for 384 experts across 40 layers. Attention, shared experts, vision, Engram lookup tables, MTP/DSpark, and other excluded components retain their source precision, including MXFP8 where applicable.

All 16,986,931,200 weight blocks passed lossless conversion, preserving dequantized weight values with signed-zero canonicalization. Activation scales were calibrated; 18 of 46,080 projection entries used fallback scales. Lossless weight conversion does not imply identical inference outputs.

The source experts already use four-bit weights. The finer NVFP4 scale layout increases checkpoint size from approximately 476 GiB to 492 GiB. The export contains 48 safetensors shards.

## Usage

### Deploy with SGLang

Use `lmsysorg/sglang:dev-cu13-dsv41` on four GB300 GPUs. The tested build was SGLang commit `da64c5cbb8cf6bfd39be19da43573fdfd484c43a`. This configuration passed loading, generation, reasoning and tool-call parsing, and image-input smoke tests.

```bash
python -m sglang.launch_server \
    --model-path nvidia/DeepSeek-V4.1-Flash-NVFP4 \
    --served-model-name DeepSeek-V4.1-Flash-NVFP4 \
    --host 127.0.0.1 --port 30000 \
    --tp 4 \
    --context-length 1048576 \
    --reasoning-parser deepseek-v41 \
    --tool-call-parser deepseekv41 \
    --chunked-prefill-size 4096 \
    --max-running-requests 16 \
    --log-level info
```

The tested build automatically selects `flashinfer_trtllm_routed` for the NVFP4 experts. Include a request-level `reasoning_effort`, such as `"max"`, to enable thinking; thinking is off by default in this SGLang build. During staging, substitute the local checkpoint directory for the repository ID if needed.

### Deploy with vLLM

Use `vllm/vllm-openai:deepseekv41-flash-0909` on four GB300 GPUs. This example adapts the recorded text-serving command to the repository ID. During staging, substitute the local checkpoint directory if needed.

```bash
vllm serve nvidia/DeepSeek-V4.1-Flash-NVFP4 \
    --tensor-parallel-size 4 \
    --data-parallel-size 1 \
    --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --language-model-only \
    --max-model-len 1048576 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}'
```

DSpark tensors are preserved, but speculative decoding was not exercised in the reported validation. The vLLM example is text-only; image serving with vLLM requires a multimodal configuration. Validate the final staged package with the intended runtime before deployment.

## Evaluation
Accuracy results obtained with vLLM are shown below, in percent.

<table>
  <tr>
    <th>Precision</th>
    <th>GPQA Diamond</th>
    <th>AA-LCR</th>
    <th>SciCode</th>
    <th>IFBench</th>
    <th>MMMU-Pro</th>
    <th>Terminal-Bench 2.1</th>
  </tr>
  <tr>
    <td>MXFP4 (source)</td>
    <td>91.035</td>
    <td>78.563</td>
    <td>54.401</td>
    <td>76.667</td>
    <td>74.046</td>
    <td>81.60</td>
  </tr>
  <tr>
    <td>NVFP4</td>
    <td>91.288</td>
    <td>78.438</td>
    <td>55.843</td>
    <td>77.267</td>
    <td>73.699</td>
    <td>82.16</td>
  </tr>
</table>

> Baseline: [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), a mixed-precision checkpoint with MXFP4 routed experts.
> Ordinary benchmark requests used `temperature=1.0`, `top_p=0.95`, `reasoning_effort=100`, and `max_new_tokens=262144`.
> GPQA and AA-LCR used 16 repeats; IFBench used 5.
> Terminal-Bench used a separate agentic configuration with eight trajectories per task.

## Model Limitations:
The base model was trained on data that contains toxic language and societal biases originally crawled from the internet. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts. The model may generate answers that may be inaccurate, omit key information, or include irrelevant or redundant text producing socially unacceptable or undesirable text, even if the prompt itself does not include anything explicitly offensive.

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

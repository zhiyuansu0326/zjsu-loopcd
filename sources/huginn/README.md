---
library_name: transformers
tags:
- code
- math
- reasoning
- llm
license: apache-2.0
language:
- en
pipeline_tag: text-generation
datasets:
  - tomg-group-umd/huginn-dataset
# datasets: # cannot order these nicely
# - HuggingFaceTB/smollm-corpus
# - jon-tow/starcoderdata-python-edu
# - ubaada/booksum-complete-cleaned
# - euirim/goodwiki
# - togethercomputer/RedPajama-Data-1T
# - allenai/dolma
# - bigcode/the-stack-v2-train-smol-ids
# - bigcode/starcoderdata
# - m-a-p/Matrix
# - cerebras/SlimPajama-627B
# - open-phi/textbooks
# - open-phi/textbooks_grounded
# - open-phi/programming_books_llama
# - nampdn-ai/tiny-strange-textbooks
# - nampdn-ai/tiny-textbooks
# - nampdn-ai/tiny-code-textbooks
# - nampdn-ai/tiny-orca-textbooks
# - SciPhi/textbooks-are-all-you-need-lite
# - vikp/textbook_quality_programming
# - EleutherAI/proof-pile-2
# - open-web-math/open-web-math
# - biglam/blbooks-parquet
# - storytracer/LoC-PD-Books
# - GAIR/MathPile
# - tomg-group-umd/CLRS-Text-train
# - math-ai/AutoMathText
# - bigcode/commitpackft
# - bigcode/stack-dedup-python-fns
# - vikp/python_code_instructions_filtered
# - mlabonne/chessllm
# - Waterhorse/chess_data
# - EleutherAI/lichess-puzzles
# - chargoddard/WebInstructSub-prometheus
# - Locutusque/hercules-v5.0
# - nvidia/OpenMathInstruct-1
# - meta-math/MetaMathQA
# - m-a-p/CodeFeedback-Filtered-Instruction
# - nvidia/Daring-Anteater
# - nvidia/sft_datablend_v1
# - BAAI/Infinity-Instruct
# - anthracite-org/Stheno-Data-Filtered
# - Nopm/Opus_WritingStruct
# - xinlai/Math-Step-DPO-10K
# - bigcode/self-oss-instruct-sc2-exec-filter-50k
# - HuggingFaceTB/everyday-conversations
# - hkust-nlp/gsm8k-fix
# - HuggingFaceH4/no_robots
# - THUDM/LongWriter-6k
# - THUDM/webglm-qa
# - AlgorithmicResearchGroup/ArXivDLInstruct
# - allenai/tulu-v2-sft-mixture-olmo-4096
# - bigscience/P3
# - Gryphe/Sonnet3.5-SlimOrcaDedupCleaned
# - Gryphe/Opus-WritingPrompts
# - nothingiisreal/Reddit-Dirty-And-WritingPrompts
# - nothingiisreal/Kalomaze-Opus-Instruct-25k-filtered
# - internlm/Lean-Github
# - pkuAI4M/LeanWorkbook
# - casey-martin/multilingual-mathematical-autoformalization
# - AI4M/leandojo-informalized
# - casey-martin/oa_cpp_annotate_gen
# - l3lab/ntp-mathlib-instruct-st
# - ajibawa-2023/Maths-College
# - ajibawa-2023/Maths-Grade-School
# - ajibawa-2023/General-Stories-Collection
# - XinyaoHu/AMPS_mathematica
# - XinyaoHu/AMPS_khan
# - Magpie-Align/Magpie-Pro-MT-300K-v0.1
# - Magpie-Align/Magpie-Reasoning-150K
# - gair-prox/FineWeb-pro
# - gair-prox/c4-pro
# - gair-prox/RedPajama-pro
# - gair-prox/open-web-math-pro
# - togethercomputer/Long-Data-Collections
# - emozilla/pg19
# - MathGenie/MathCode-Pile
# - KingNish/reasoning-base-20k
# - nvidia/OpenMathInstruct-2
# - LLM360/TxT360
# - neuralwork/arxiver
---

# Huginn-0125
This is Huginn, version 01/25, a latent recurrent-depth model with 3.5B parameters, trained for 800B tokens on AMD MI250X machines. This is a proof-of-concept model, but surprisingly capable in reasoning and code given its training budget and size.
All details on this model can be found in the tech report: "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach." (https://www.arxiv.org/abs/2502.05171)
For more information, see the paper page: https://huggingface.co/papers/2502.05171.

8 intermediate checkpoints of the model can be found in its collection. Additional intermediate checkpoints are available upon request while we find a place to host all ~350 of them. The data used to train
this model is publicly available (entirely on Hugging Face), and scripts provided with the pretraining code at https://github.com/seal-rg/recurrent-pretraining can be used to repeat our preprocessing and our entire training run. 

<img src="asset2.jpeg" width="60%">



##  Table of Contents

1. [How to Use](#downloading-and-using-the-model)
2. [Advanced Usage](#advanced-features)
3. [Model Summary](#model-summary)
4. [Limitations](#limitations)
5. [Technical Details](#training)
6. [License](#license)
7. [Citation](#citation)


## Downloading and Using the Model
Load the model like this:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
```
### Modifying the Model's Depth at Test Time:
By providing the argument `num_steps`, the model will execute a forward pass with that amount of compute: 
```python
input_ids = tokenizer.encode("The capital of Westphalia is", return_tensors="pt", add_special_tokens=True).to(device)
model.eval()
model.to(device)

model(input_ids, num_steps=32)
```
The model has about 1.5B parameters in its non-recurrent layers (prelude+coda), 0.5B parameters in the embedding, and 1.5B recurrent parameters, so, as a guideline, 
the number of materialized parameters is `num_steps * 1.5B + 2B`. Playing with this parameter is what makes this model interesting, and different from fixed-depth transformers!
The model is trained to accept an arbitrary number of steps. However, using fewer than 4 steps will result in very coarse answers. If given enough context to reason about, benchmarks show the model improving up to around `num_steps=64`. Beyond that, more steps generally do not hurt, but we see no further improvements.

*Note*: Due to an upload issue the model is currently stored on HF with 2 copies of the tied embedding, instead of just one. This will be fixed in a future release.

### Inference
The model was trained with bfloat16-mixed precision, so we recommend using `bfloat16` to run inference (or AMP bfloat16-mixed precision, if you really want). All benchmarks were evaluated in pure `bfloat16`.

### Sampling
The model can be used like a normal HF model to generate text with KV-caching working as expected. You can provide `num_steps` directly to the `generate` call, for example:
```
model.eval()
config = GenerationConfig(max_length=256, stop_strings=["<|end_text|>", "<|end_turn|>"], 
                          use_cache=True,
                          do_sample=False, temperature=None, top_k=None, top_p=None, min_p=None, 
                          return_dict_in_generate=True,
                          eos_token_id=65505,bos_token_id=65504,pad_token_id=65509)


input_ids = tokenizer.encode("The capital of Westphalia is", return_tensors="pt", add_special_tokens=True).to(device)
outputs = model.generate(input_ids, config, tokenizer=tokenizer, num_steps=16)
```

*Note*: `num_steps` and other model arguments CANNOT be included in the `GenerationConfig`, they will shadow model args at runtime.


### Chat Templating

The model was not finetuned or post-trained, but due to inclusion of instruction data during pretraining, natively understand its chat template. You can chat with the model like so
```
messages = []
messages.append({"role": "system", "content" : "You are a helpful assistant."})
messages.append({"role": "user", "content" : "What do you think of Goethe's Faust?"})
chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(chat_input)
input_ids = tokenizer.encode(chat_input, return_tensors="pt", add_special_tokens=False).to(device)

model.generate(input_ids, config, num_steps=64, tokenizer=tokenizer)
```

### KV-cache Details
The model requires its own KV-cache implementation `HuginnDynamicCache`, otherwise the KV-caches of later calls to the recurrent block will overwrite the earlier ones.
The current implementation will always try to inject this Cache implementation, but that may break with huggingface updates. If you do not use generate, but implement your own generation, use a pattern like this:

```python
# first step:
past_key_values = None
outputs = model(input_ids=input_ids, use_cache=True, past_key_values=past_key_values)
past_key_values = outputs.past_key_values # Should be an instance of HuginnDynamicCache
# next step
outputs = model(input_ids=input_ids, use_cache=True, past_key_values=past_key_values)
```

## Advanced Features

### Per-Token Adaptive Compute
When generating, you can use a variable amount of compute per-token. The model is not trained for this, so this is a proof-of-concept, that it can do this task zero-shot. 
You can pick between a few sane stopping rules, `entropy-diff`, `latent-diff`,`kl` and `argmax-stability`, via `criterion=...`. The exit threshold can be modified via `exit_threshold=5e-4`.
We suggest using `kl` for interesting exits and `argmax-stability` for conservative exits. Note that using these variables overrides the default generation function. Not all arguments that are valid for the normal `generate` call are valid here. To make this more explicit, you can also directly call `generate_with_adaptive_compute`:

```python
from transformers import TextStreamer
streamer = TextStreamer(tokenizer)

model.generate_with_adaptive_compute(input_ids, config, num_steps=64, tokenizer=tokenizer, streamer=streamer,
                                     continuous_compute=False, criterion="kl", exit_threshold=5e-4, cache_kwargs={"lookup_strategy": "latest-m4"})

```
Your cache strategy should be set to `"latest-m4"` if using adaptive compute.

### KV-cache Sharing
To reduce KV cache memory requirements, the model can be run with fewer KV-caches, with later iterations in the recurrence overwriting earlier caches. To use this feature, set
the cache argument `lookup_strategy` to include `compress-s16` (where the last number determine the size of the cache).
```
model.generate_with_adaptive_compute(input_ids, config, num_steps=64, tokenizer=tokenizer, streamer=streamer,
                                     continuous_compute=False, cache_kwargs={"lookup_strategy": "compress-s16"})
```
You can combine this per-token adaptive compute. In that case your lookup strategy should be `latest-m4-compress-s16`.

### Warmstart / Continuous CoT
At each generation step, the recurrence can be warmstarted with the final state from the previous token by setting `continuous_compute=True`, like so
```
model.generate_with_adaptive_compute(input_ids, config, num_steps=64, tokenizer=tokenizer, streamer=streamer, continuous_compute=True)
```



## Model Summary
The model is primarily structured around decoder-only transformer blocks. However these blocks are structured into three functional groups, the __prelude__ \\(P\\), 
which embeds the input data into a latent space using multiple transformer layers, then the core __recurrent block__ \\(R\\), which is the central unit of recurrent 
computation modifying states \\(\mathbf{s} \in \mathbb{R}^{n \times h }\\), and finally the __coda__ \\(C\\), which un-embeds from latent space using several layers and
also contains the prediction head of the model. 

Given a number of recurrent iterations \\(r\\), and a sequence of input tokens \\(\mathbf{x} \in V^n\\) these groups are used in the following way to produce output 
probabilities \\(\mathbf{p} \in \mathbb{R}^{n \times |V|}\\).

$$\mathbf{e} = P(\mathbf{x})$$

$$\mathbf{s}_0 \sim \mathcal{N}(\mathbf{0}, \sigma^2 I_{n\cdot h})$$

$$\mathbf{s}_i = R(\mathbf{e}, \mathbf{s}_{i-1}) \; \textnormal{for} \;  i \in \lbrace 1, \dots, r \rbrace$$

$$\mathbf{p} = C(\mathbf{s}_r)$$
where \\(\sigma\\) is the standard deviation of the initial random state. Given an init random state \\(\mathbf{s}_0\\), the model repeatedly applies the core recurrent 
block \\(R\\), which accepts the latent state \\(\mathbf{s}_{i-1}\\) and the embedded input \\(\mathbf{e}\\) and outputs a new latent state \\(\mathbf{s}_i\\). 
After finishing all iterations, the coda block processes the last state and produces the probabilities of the next token.

Please refer to the paper for benchmark performance on standard benchmarks.

## Limitations
Our checkpoint is trained for only 47000 steps on a broadly untested data mixture with a constant learning rate. As an academic project, the model is trained only on publicly available data and the 800B token count, while large in comparison to older fully open-source models such as the Pythia series, is small in comparison to modern open-source efforts such as OLMo, and tiny in comparison to the datasets used to train industrial open-weight models.

## Technical Specifications
This model was trained on 21 segments of 4096 AMD MI-250X GPUs on the OLCF Frontier Supercomputer in early December 2024. The model was trained using ROCM 6.2.0, and PyTorch 2.6 nightly pre-release 24/11/02. The code used to train the model can be found at https://github.com/seal-rg/recurrent-pretraining.

## License
This model is released under the [apache-2.0](https://choosealicense.com/licenses/apache-2.0/) licence.

## Citation
```
@article{geiping_scaling_2025,
  title = {Scaling up {{Test-Time Compute}} with {{Latent Reasoning}}: {{A Recurrent Depth Approach}}},
  shorttitle = {Scaling up {{Test-Time Compute}} with {{Latent Reasoning}}},
  author = {Geiping, Jonas and McLeish, Sean and Jain, Neel and Kirchenbauer, John and Singh, Siddharth and Bartoldson, Brian R. and Kailkhura, Bhavya and Bhatele, Abhinav and Goldstein, Tom},
  year = {2025},
  month = feb,
  eprint = {2502.05171},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2502.05171},
  url = {http://arxiv.org/abs/2502.05171},
  urldate = {2025-02-10},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Computation and Language,Computer Science - Machine Learning},
  journal = {arxiv:2502.05171[cs]}
}
```

## Contact
Please, feel free to contact us with any questions, or open a discussion thread on Hugging Face.
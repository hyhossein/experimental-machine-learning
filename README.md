# Experimental Machine Learning

**A Practical Reference for ML Engineers**

*From Model Fundamentals to Production Deployment*

April 2026

---

## 1. What a Model Actually Is

A trained model is a frozen snapshot of a mathematical function found by optimization. It maps sequences of tokens to probability distributions over the next token. The entire intelligence of the model lives in billions of floating-point numbers (weights) organized as tensors across layers.

The architecture (transformer blocks, attention heads, MLPs) is the wiring diagram — it defines the order and structure of operations. The weights are the knowledge. Same wiring, different weights = different model.

> *Key insight: The model doesn't store facts or rules. It stores compressed statistical structure of its training distribution — a generative theory of how text is produced, not a database of text that was seen. This is why models generalize to inputs never encountered during training.*

### 1.1 Model File Formats

| Format | Purpose | Structure | Use Case |
|---|---|---|---|
| SafeTensors | Full-precision storage | JSON header + raw binary weights, memory-mapped | Training, fine-tuning, GPU inference |
| GGUF | Quantized deployment | Binary header with full metadata + quantized weights | CPU/edge inference via llama.cpp |

> *SafeTensors is a storage format for the training ecosystem. GGUF is a deployment format for the inference ecosystem. SafeTensors stores exact coordinates; GGUF stores a lossy sketch.*

### 1.2 The Forward Pass

Every inference step follows the same sequence: token IDs are looked up in the embedding table to become vectors. Each vector passes through N transformer layers (attention + MLP). The final layer projects to vocabulary-sized logits. Softmax converts logits to probabilities. A token is sampled.

Every @ (matrix multiply) in the forward pass uses the stored weight tensors. The code is identical across all transformer models — the weights are what differentiate them.

---

## 2. Inference-Time Steering

These techniques modify the model's output without changing any weights. The model "thinks" the same thing; you reshape which tokens get chosen.

### 2.1 Sampling Parameters

| Parameter | What It Does | When to Use |
|---|---|---|
| Temperature | Scales logits before softmax. <1 = sharper (deterministic), >1 = flatter (creative) | Low for factual tasks, high for creative generation |
| Top-p | Samples from smallest set whose cumulative probability exceeds p | Adaptive: decisive when model is sure, exploratory when uncertain |
| Top-k | Samples only from the k highest-probability tokens | Simpler alternative to top-p, less adaptive |
| Logit bias | Adds/subtracts from specific token logits before softmax | Suppress unwanted words, boost desired vocabulary |
| Repetition penalty | Reduces logit of already-generated tokens | Prevents degenerate loops in autoregressive generation |

> *Repetition penalty exists because autoregressive generation has a feedback loop: once a phrase appears, it enters the context and becomes more likely, which reinforces it further. The penalty breaks this cycle.*

### 2.2 Classifier-Free Guidance (CFG)

Run the model twice: once with your prompt, once without (null prompt). Amplify the difference. This isolates what your specific prompt contributed versus generic model behavior, making output more faithful to intent. Borrowed from diffusion models.

**Best for:** situations where the model isn't leaning hard enough into your specific instructions.

### 2.3 Self-Consistency

Run the same prompt N times at temperature > 0. Take the majority vote. Different runs explore different reasoning paths through the model's distribution. Cheap to implement, typically gains 3–8% on reasoning tasks.

---

## 3. Activation-Level Intervention

Modify the model's internal representations mid-computation without changing weights.

### 3.1 Activation Steering via Contrastive Pairs

The process:

1. **Step 1:** Create ~50 prompt pairs that differ in exactly one behavioral dimension (e.g., honest vs. sycophantic responses to the same question).
2. **Step 2:** Run all prompts through the model. At a chosen layer, extract the hidden state vectors (e.g., 4096-dim at layer 16).
3. **Step 3:** Average the honest vectors (H), average the sycophantic vectors (S). Compute H − S. This difference vector is the steering direction.
4. **Step 4:** At inference time, add this direction vector to the hidden states at that layer for any new input.

> *This works because transformer representations organize behavioral traits as linear directions in activation space. The model already has an internal "dial" for honesty, formality, verbosity — you're just reaching in and turning it. Limitation: complex, context-dependent behaviors don't reduce to a single direction.*

**Quality depends on pair quality.** If your "honest" responses also happen to be shorter, you've contaminated the direction with a brevity component. Isolate exactly one trait per experiment.

---

## 4. Parameter-Efficient Fine-Tuning

### 4.1 LoRA

Freeze the original weights W. Add a low-rank correction ΔW = AB where A is (d × r) and B is (r × d), with r << d (typically 8–16). Train only A and B via standard backpropagation.

> *Why low-rank works: useful adaptations concentrate along a handful of directions in weight space. The bottleneck isn't a limitation — it's a discovery about adaptation structure. A rank-16 correction to a 4096×4096 matrix means the adaptation lives in a 16-dimensional subspace.*

### 4.2 QLoRA

Quantize the frozen base model to 4-bit (NF4). Attach LoRA adapters in full precision (fp16/bf16). Base weights are compressed and frozen; adapters learn in full precision. Dequantize on-the-fly during forward pass.

**Impact:** Train a 70B model on a single 48GB GPU. Base model ~35GB in 4-bit, adapters a few hundred MB.

NF4 (NormalFloat 4-bit) spaces quantization levels according to the normal distribution, giving more precision near zero where most weights cluster. Double quantization compresses the quantization constants themselves, saving another ~0.4 bits per parameter.

### 4.3 Other LoRA Variants

| Variant | Key Idea | Advantage |
|---|---|---|
| DoRA | Decompose weight into magnitude + direction, adapt separately | Better matches full fine-tuning dynamics |
| LoRA+ | Different learning rates for A and B matrices | B (projection back) learns slower; simple change, measurable gain |
| AdaLoRA | Adaptive rank per layer during training | Automatically allocates capacity where gradients are strongest |
| GaLore | Project gradients into low-rank subspace instead of weights | Memory savings without architectural modification |

---

## 5. Quantization

Map float16 weights to fewer bits. For each block of weights (32–128), compute scale and zero-point, store each weight as a small integer relative to that scale. Reconstruction: weight = min + quant × scale / (levels − 1).

### 5.1 Methods Compared

| Method | Strategy | Tradeoff |
|---|---|---|
| Naive round-to-nearest | Quantize all weights uniformly | Fast but loses quality on outlier-heavy layers |
| GPTQ | Sequential error correction: quantize one weight, use its error to adjust the next | Slow calibration, good quality |
| AWQ | Scale critical weights up before quantization (based on activation magnitude) | Fast calibration, comparable quality to GPTQ |
| NF4 (QLoRA) | Non-uniform levels spaced by normal distribution | Designed for weight distributions, used in QLoRA |

> *Not all weights are equally sensitive. AWQ discovered that ~1% of weights are critical — protecting those at higher precision while aggressively compressing the rest retains 95%+ model quality. The function's behavior is dominated by a small subset of its parameters.*

**Practical note:** Q4 quantization on GPU is often slower than fp16 for small batch sizes because dequantization overhead can exceed the memory bandwidth savings. Profile before committing.

---

## 6. Inference Optimization

### 6.1 KV Cache

During autoregressive generation, each new token must attend to all previous tokens. Without caching, this requires recomputing Keys and Values for every previous token at every step (O(N²)). The KV cache stores these, reducing generation to O(N). For a 70B model at 128K context, the cache alone can consume 40+ GB of VRAM.

- **GQA (Grouped-Query Attention):** Groups of heads share K/V to reduce cache size. Less expressive but massive memory savings.
- **Sliding Window:** Only attend to last N tokens. Cache stays bounded. Information propagates indirectly through layers.

### 6.2 Speculative Decoding

Use a small, fast model (draft) to generate K candidate tokens. Run one forward pass of the large model (verifier) on all K candidates simultaneously. The large model checks each position: "Would I have produced this token?" Accept all tokens up to the first disagreement. Resample from the large model at the point of disagreement.

> *Zero quality loss — mathematically guaranteed to produce the same distribution as the large model alone. You're exploiting the fact that most tokens are "easy" and don't need the full model. Only surprising tokens require the big model to actually think. Typical speedup: 2–3x.*

**Verification is parallel, generation is sequential.** That's the asymmetry that makes it work. Checking 8 tokens at once is one batched operation; generating 8 tokens is 8 sequential passes.

### 6.3 Mixture of Experts (MoE)

Replace the single FFN per layer with N expert FFNs plus a tiny router. The router (a small linear layer) takes each token's hidden state, produces N scores, and activates the top-K experts. Only K out of N experts compute per token.

**Example:** Mixtral has 47B total parameters but only uses ~13B per token (2 of 8 experts). Quality of a larger model at inference cost of a smaller one.

Experts develop statistical specializations through training — not clean human-labeled categories. A load-balancing loss prevents router collapse (one expert attracting all tokens via rich-get-richer dynamics).

---

## 7. Model Merging

Combine capabilities from separately fine-tuned models without retraining. Only works reliably when models share a common ancestor (aligned internal coordinate systems).

### 7.1 Methods

| Method | Mechanism | When to Use |
|---|---|---|
| Linear average | (W_A + W_B) / 2 | Quick baseline, no conflict resolution |
| SLERP | Spherical interpolation preserving magnitude | Better than linear when weight magnitudes matter |
| Task arithmetic | Add task vectors: base + (W_A − base) + (W_B − base) | Combine independent capabilities additively |
| TIES | Trim noise, elect sign on conflicts, merge agreements | Handles conflicting fine-tunes intelligently |
| DARE | Drop 90% of task vector entries randomly, rescale | Reduces interference before merging |
| DARE-TIES | DARE sparsification then TIES conflict resolution | Current best practice for production merging |

> *Two independently trained models may represent the same concept using completely different internal coordinates. Merging them produces nonsense. Models fine-tuned from the same base share coordinate systems because they didn't move far from the starting point.*

### 7.2 Production Protocol

1. Always start from a shared base model for all fine-tunes you plan to merge.
2. Use LoRA adapters rather than full fine-tuning — smaller deltas mean less interference.
3. Apply DARE first (sparsify), then TIES (resolve conflicts).
4. Evaluate per-task metrics after merging, not just aggregate. Merging can silently degrade one capability while improving another.
5. Consider adapter stacking at inference time (base + α·LoRA_A + β·LoRA_B) for controllable, debuggable blends.

---

## 8. Alignment: From Predictor to Assistant

A pretrained model is a next-token predictor with no concept of "should." Alignment adds behavioral constraints.

### 8.1 The Pipeline

- **SFT (Supervised Fine-Tuning):** Train on curated assistant conversations. Teaches the format of being helpful. Limited by example coverage.
- **RLHF:** Human raters rank outputs → train a reward model → use PPO to maximize reward. Powerful but unstable: three models training simultaneously, reward hacking risk, expensive.
- **DPO (Direct Preference Optimization):** Skip the reward model entirely. Directly optimize on preference pairs (preferred vs. dispreferred response). Same results as RLHF with dramatically less complexity. One training loop.
- **KTO:** Needs only thumbs-up/thumbs-down, not paired comparisons. Grounded in prospect theory (loss aversion). Cheapest data collection.
- **Constitutional AI:** Model critiques its own output against written principles, generates revision pairs, then trains on those via DPO. Recursive self-improvement bounded by explicit rules. More interpretable and auditable than RLHF.

---

## 9. Architecture Concepts

### 9.1 Attention as Dynamic Routing

Each token produces Query (what am I looking for?), Key (what do I contain?), Value (what do I offer?). Dot product of Q and K produces relevance scores. Softmax normalizes them. Output is a weighted sum of Values.

> *Attention lets the model rewire its own computation graph based on the input. Different input, different routing. This is why transformers outperform earlier architectures — they build input-dependent computation paths. Multi-head attention runs parallel routing in different subspaces simultaneously.*

### 9.2 The Residual Stream

Residual connections (x = x + layer_output) create a shared communication bus that all layers read from and write to additively. Early layers contribute syntax, middle layers semantics, late layers task-specific computation. This additive structure is why linear interventions (steering vectors, LoRA) work — your intervention is just one more additive term in a sum of many.

---

## 10. Deep Research Insights

- **Lottery Ticket Hypothesis:** 90%+ of parameters are search scaffolding. The actual solution is a tiny subnetwork. Overparameterization isn't waste — it's room to maneuver during training. This is why pruning and LoRA work.
- **Grokking:** Memorization happens fast. True understanding emerges much later, invisibly, while training loss is already zero. You cannot distinguish memorization from generalization using training metrics.
- **Double Descent:** Test error vs. model size is non-monotonic. There's a phase transition where overfitting peaks, then bigger models suddenly generalize better. Behaves like a physical phase change.
- **Superposition:** Models store millions of concepts in thousands of dimensions via nearly-orthogonal directions. Every neuron encodes hundreds of features. Information is holographic — smeared everywhere, nothing local.
- **Scaling Laws:** Performance follows power laws in compute, data, and parameters. Predictive across orders of magnitude. You can forecast model quality before training it. Chinchilla-optimal: ~20 tokens per parameter, but inference economics favor overtraining smaller models.
- **Loss ≠ Capability:** Capabilities emerge discontinuously. Tiny loss improvements unlock entirely new abilities. Two models with identical loss can have completely different skill profiles.
- **The Bitter Lesson (amended):** Scale beats cleverness, except when the right inductive bias buys 10x efficiency. The real skill is predicting which biases survive scaling. Attention survived. Everything before it didn't.

---

## 11. Knowledge Distillation Protocol

Train a small model to mimic a large model's outputs using stored inference results. Zero additional API cost if you've already collected the large model's reasoning.

### 11.1 Steps

1. **Collect:** Run the large model (teacher) on your task. Store full reasoning chains plus final answers.
2. **Format:** Convert to training pairs: input = question/prompt, label = teacher's chain-of-thought + answer.
3. **Train:** QLoRA fine-tune the small model (student) on these pairs. The student learns the teacher's reasoning patterns, not just its answers.
4. **Evaluate:** Compare distilled student vs. prompted student vs. teacher on held-out data.

> *Distillation extracts value from API calls you've already paid for. A 7B model trained on a frontier model's reasoning chains can close 50–70% of the accuracy gap at zero marginal inference cost.*

---

## 12. Decision Framework: Choosing the Right Technique

| Goal | First Try | If Insufficient | Nuclear Option |
|---|---|---|---|
| Better accuracy | Better prompts + RAG | Self-consistency / ensemble | QLoRA fine-tuning |
| Faster inference | Quantization (AWQ/GPTQ) | Speculative decoding | Smaller model + distillation |
| Lower memory | Quantization (Q4/Q8) | GQA + sliding window | MoE architecture |
| Domain adaptation | RAG + few-shot examples | QLoRA on domain data | Continual pretraining |
| Behavioral control | System prompt + logit bias | Activation steering | DPO on preference data |
| Combine capabilities | Adapter stacking (α·LoRA_A + β·LoRA_B) | DARE-TIES merging | Multi-task fine-tuning from scratch |
| Reduce cost | Smaller quantized model | Distillation | MoE + speculative decoding |

Always start with the cheapest intervention. Move right only when measured results are insufficient. Each column costs roughly 10x more in compute and engineering time.

---

## 13. Quick Reference: The Intervention Spectrum

From lightest to heaviest, every technique falls on this spectrum:

| Level | What Changes | Cost | Reversible? |
|---|---|---|---|
| Prompt engineering | The function's input | Zero | Instantly |
| Sampling parameters | The output distribution | Zero | Instantly |
| Logit biases | Individual token probabilities | Zero | Instantly |
| Activation steering | Mid-computation hidden states | Minimal (contrastive pair collection) | Instantly (additive) |
| LoRA / QLoRA | Small learned weight corrections | Hours of GPU | Detachable adapters |
| Model merging | Weight-space combination | Minutes of compute | Irreversible (baked in) |
| Full fine-tuning | All weights | Days of GPU | Irreversible |
| Continued pretraining | Foundation knowledge | Weeks of GPU | Irreversible |

> *The most surprising insight across all of this: for most use cases, you never need to touch the weights. The function is general enough that changing its input is sufficient to radically change its behavior. That's the power of a good next-token predictor — it implicitly models every persona, domain, and style. You just invoke the right one through context.*

---

## 14. Inference-Time Compute Scaling

The traditional scaling paradigm focuses on training: more parameters, more data, more FLOPs during the learning phase. Inference-time compute scaling inverts this — spend more compute *at generation time* to get better outputs from a fixed model. The model stays the same; you give it more room to think.

### 14.1 Chain-of-Thought and Extended Reasoning

Prompting a model to "think step by step" is the simplest form of inference-time scaling. Each intermediate token the model generates becomes part of its context, effectively extending the computation depth beyond what the architecture alone provides. The model's fixed forward pass is shallow (dozens of layers), but autoregressive chain-of-thought turns it into a deep computation by routing intermediate results back through the full network at each step.

> *The key insight: a transformer's forward pass has constant depth (number of layers), but chain-of-thought makes the effective computation depth proportional to the number of generated tokens. Each reasoning step is a full forward pass that conditions on all prior steps. This is why "think harder" actually works — it's not a metaphor, it's literally more computation.*

### 14.2 Best-of-N and Majority Voting

Generate N completions independently, then select the best one. Selection can be via majority vote (most common final answer), reward model scoring, or verifier-based filtering. This exploits the stochastic nature of sampling — the model's distribution contains better answers than its single greedy output. Scaling N produces log-linear improvements on reasoning benchmarks.

### 14.3 Tree Search and Verification

Structure generation as a search problem. At each step, expand multiple candidate continuations, evaluate partial solutions with a verifier, and prune bad branches early. This is the core pattern behind systems like AlphaCode and o1-style reasoning — the model proposes, a verifier disposes, and search navigates the space between them.

Verification is the bottleneck. A good verifier (learned reward model, unit test suite, formal proof checker) makes tree search powerful. Without one, you're doing expensive random sampling.

### 14.4 Adaptive Compute Allocation

Not all queries require the same amount of inference-time compute. Simple factual lookups need one forward pass. Multi-step mathematical proofs need hundreds of reasoning tokens. The frontier is routing different queries to different compute budgets dynamically — spending inference FLOPs where they matter.

| Strategy | Mechanism | Compute Scaling |
|---|---|---|
| Greedy decoding | Single forward pass per token | 1x (baseline) |
| Chain-of-thought | Extended reasoning in context | ~5–50x per query |
| Best-of-N | N independent completions | Nx linear |
| Tree search + verifier | Branching with pruning | Variable, guided by verifier |
| Iterative refinement | Generate → critique → revise loops | Kx per revision cycle |

> *The economics of inference-time scaling are different from training-time scaling. Training compute is paid once and amortized across all queries. Inference compute is paid per query. This means inference-time techniques are most valuable for high-stakes, low-volume tasks (code generation, theorem proving, medical diagnosis) where the cost of a wrong answer exceeds the cost of extra compute.*

---

## 15. Autonomous Kernel Optimization (AutoKernel)

Most GPU workloads run at a fraction of theoretical hardware throughput. The gap between what a GPU can do and what production code actually achieves is enormous — typical utilization sits around 15–20% for unoptimized kernels. AutoKernel applies the autoresearch paradigm to GPU kernel optimization: an AI agent autonomously discovers hardware-specific Triton kernels that close this gap.

### 15.1 The Autoresearch Loop

The core pattern is borrowed from Karpathy's autoresearch concept: an agent modifies a single file, runs a fixed evaluation, keeps or reverts the change, and repeats indefinitely. Applied to GPU kernels, this becomes:

1. **Profile** a PyTorch model to identify computational bottlenecks by GPU time.
2. **Extract** each bottleneck as a standalone Triton kernel with a PyTorch reference implementation.
3. **Optimize** each kernel autonomously — the agent edits the kernel, a benchmark harness runs 5-stage correctness checks (smoke test, shape sweep, numerical stability, determinism, edge cases) plus performance measurement against roofline, and the change is kept only if it passes all checks and improves throughput.
4. **Verify** end-to-end by plugging optimized kernels back into the original model.

Each experiment cycle takes roughly 90 seconds, yielding ~40 experiments per hour or ~320 overnight across all kernels. The orchestrator uses Amdahl's law to prioritize: a modest speedup on a kernel consuming 60% of GPU time beats a dramatic speedup on a kernel consuming 5%.

### 15.2 Why Triton

Triton provides Python-like syntax that AI agents can read, modify, and reason about without mastering inline PTX or SASS. Well-tuned Triton regularly reaches 80–95% of cuBLAS performance. The compile-in-seconds iteration speed is critical — the agent needs fast feedback loops to explore hundreds of optimization strategies overnight.

### 15.3 Correctness-First Design

A fast but wrong kernel is immediately reverted. The benchmark checks kernel output against the PyTorch reference before measuring performance. This prevents the agent from "optimizing" by producing numerically incorrect results — a failure mode that would be invisible without systematic verification.

### 15.4 Production Kernel Optimization (Forge)

Forge extends this concept into a production service: given any AI model and a target GPU (H100, A100, L40S, B200), it generates optimized CUDA/Triton kernels as drop-in replacements with verified numerical correctness. The claimed results include up to 3x inference speedup and GPU utilization improvements from ~16% to ~88%, translating directly into infrastructure cost savings.

The key insight is that kernel optimization is hardware-specific. A kernel tuned for H100's memory hierarchy performs differently on A100. Generating per-hardware optimized kernels rather than using generic implementations captures performance that one-size-fits-all libraries leave on the table.

---

## 16. Evolutionary Code Optimization (OpenEvolve)

OpenEvolve is an open-source implementation of the AlphaEvolve concept: using LLMs as mutation operators inside an evolutionary algorithm to discover optimized code. Rather than a human writing better code through insight, an LLM proposes modifications, an evaluator scores them, and evolution selects for fitness — running continuously until convergence or budget exhaustion.

### 16.1 MAP-Elites + LLM Mutations

The core algorithm combines MAP-Elites (a quality-diversity evolutionary method) with LLM-generated code mutations:

1. **Population**: A grid of programs organized by feature dimensions (code complexity, performance, structural diversity). Each cell holds the best program found for that combination of features.
2. **Selection**: Pick parent programs — a mix of top performers (exploitation) and diverse programs from underexplored cells (exploration).
3. **Mutation**: An LLM receives the parent code plus execution feedback (errors, profiling data, previous attempts) and proposes modifications. This is where the LLM's knowledge of algorithms and programming patterns acts as an informed mutation operator, far more targeted than random perturbation.
4. **Evaluation**: Run the mutated code through a user-defined evaluator that returns scores and artifacts (stderr, profiling data, correctness checks).
5. **Insertion**: If the mutant scores well for its feature coordinates, it replaces the current occupant of that MAP-Elites cell.

### 16.2 Island Model and Migration

Multiple populations (islands) evolve independently with periodic migration of high-fitness programs between them. This prevents premature convergence — different islands explore different regions of the solution space, and migration introduces successful strategies across populations.

### 16.3 Demonstrated Results

The approach has produced concrete outcomes across domains: state-of-the-art results on circle packing (n=26), 2–3x GPU kernel speedups on Apple Silicon via Metal shader evolution, automated discovery of adaptive sorting algorithms in Rust, and prompt optimization achieving +23% accuracy on HotpotQA.

### 16.4 The Artifact Feedback Loop

Execution artifacts (error messages, profiling data, build warnings) are fed back into the next generation's LLM prompt. This creates a directed search: the LLM doesn't just mutate randomly — it reads what went wrong and proposes targeted fixes. Each generation learns from the failures of the previous one, mimicking the debugging process a human programmer would follow but running at machine speed.

> *The deeper pattern: evolutionary algorithms have always been limited by the quality of their mutation operators. Random bit flips or crossover produce mostly garbage. LLMs as mutation operators change this fundamentally — they bring domain knowledge, programming conventions, and optimization intuition to each proposed change. The mutation operator itself is intelligent.*

---

## 17. References

### Papers and Foundational Research

- Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized Language Models.* arXiv:2305.14314.
- Lin, J., et al. (2023). *AWQ: Activation-aware Weight Quantization.* arXiv:2306.00978.
- Frantar, E., et al. (2022). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* arXiv:2210.17323.
- Leviathan, Y., et al. (2023). *Fast Inference from Transformers via Speculative Decoding.* ICML.
- Shazeer, N., et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR.
- Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS.
- Ethayarajh, K., et al. (2024). *KTO: Model Alignment as Prospect Theoretic Optimization.* arXiv:2402.01306.
- Bai, Y., et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* arXiv:2212.08073.
- Frankle, J. & Carlin, M. (2019). *The Lottery Ticket Hypothesis.* ICLR.
- Power, A., et al. (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* arXiv:2201.02177.
- Elhage, N., et al. (2022). *Toy Models of Superposition.* Anthropic.
- Kaplan, J., et al. (2020). *Scaling Laws for Neural Language Models.* arXiv:2001.08361.
- Hoffmann, J., et al. (2022). *Training Compute-Optimal Large Language Models (Chinchilla).* arXiv:2203.15556.
- Sutton, R. (2019). *The Bitter Lesson.* Blog post.
- Yadav, P., et al. (2023). *TIES-Merging: Resolving Interference When Merging Models.* NeurIPS.
- Yu, L., et al. (2023). *Language Model Fusion via DARE.* arXiv:2311.03099.
- Turner, A., et al. (2023). *Activation Addition: Steering Language Models Without Optimization.* arXiv:2308.10248.
- Liu, S., et al. (2023). *DoRA: Weight-Decomposed Low-Rank Adaptation.* arXiv:2402.09353.
- Zhao, J., et al. (2024). *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection.* arXiv:2403.03507.
- Wang, X., et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR.
- Snell, C., et al. (2024). *Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters.* arXiv:2408.03314.

### Tools, Projects, and Platforms

- **AutoKernel** — Autoresearch for GPU kernels. Autonomous Triton kernel optimization using the edit-benchmark-keep/revert loop. Inspired by Karpathy's autoresearch.
  - Repository: [github.com/RightNow-AI/autokernel](https://github.com/RightNow-AI/autokernel)
  - License: MIT

- **Forge** (RightNow AI) — Production GPU kernel optimization service. Generates hardware-specific optimized CUDA/Triton kernels as drop-in replacements with verified numerical correctness. Supports NVIDIA datacenter GPUs (H100, A100, B200, L40S).
  - Product page: [rightnowai.co/forge](https://www.rightnowai.co/forge)

- **OpenEvolve** — Open-source evolutionary coding agent implementing the AlphaEvolve concept. Uses LLMs as mutation operators inside MAP-Elites evolutionary search to discover optimized algorithms autonomously.
  - Repository: [github.com/algorithmicsuperintelligence/openevolve](https://github.com/algorithmicsuperintelligence/openevolve)
  - License: Apache-2.0

- **Karpathy, A.** (2025). *autoresearch* — The original experiment in autonomous AI research agents for LLM training.
  - Repository: [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)

- **Triton** (OpenAI) — Python-like language for writing GPU kernels. Compiles to optimized GPU code with near-cuBLAS performance.
  - Repository: [github.com/triton-lang/triton](https://github.com/triton-lang/triton)

- **llama.cpp** (Gerganov, G.) — GGUF inference runtime for CPU and edge deployment.
  - Repository: [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

- **Hugging Face PEFT** — Library for parameter-efficient fine-tuning (LoRA, QLoRA, AdaLoRA, etc.).
  - Repository: [github.com/huggingface/peft](https://github.com/huggingface/peft)

- **vLLM** — High-throughput LLM serving engine with PagedAttention.
  - Repository: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

---


## 18. Production GPU Deployment on Azure ML (SDK v2)

Everything up to this point assumes you have a model and want to make it better, faster, or smaller. This section assumes you have a model and need to put it behind a URL that other systems can call reliably. The gap between "it works on my machine" and "it works as a managed endpoint" is where most ML engineering time actually goes.

Azure ML SDK v2 organizes deployment around five objects: **MLClient** (authentication), **Endpoint** (the stable URL), **Deployment** (the running container behind that URL), **Environment** (the Docker image plus packages), and **CodeConfiguration** (the scoring script). Understanding the boundaries between these objects prevents most first-time deployment failures.

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="your-sub-id",
    resource_group_name="your-rg",
    workspace_name="your-workspace",
)
```

`DefaultAzureCredential` works automatically from Azure ML compute instances. Use `AzureCliCredential` after `az login` when running from a local machine.

### 18.1 The Environment Build Problem

Azure ML builds a Docker image from your conda YAML plus a base image. Complex environments — anything combining PyTorch, CUDA, transformers, and vLLM — routinely time out during the Docker build step. This is the single biggest source of failed deployments, and it happens before your code even runs.

Three principles that prevent most build failures: keep conda YAMLs minimal with exact version pins and don't mix conda channels with pip for the same library. Use Microsoft's CUDA base images (`mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:latest`) instead of building CUDA from scratch. Register the environment separately before deploying — this isolates the Docker build from the model download and container startup, so you know immediately which step failed.

When a build times out, Azure throws `ResourceOperationFailure` with "timeout while waiting for Environment Image." The actual build log lives in your workspace storage account under `aml-environment-image-build/{env-name}/{version}/`. The error message from Azure itself is never specific enough to diagnose the problem — always check the build log directly.

A working GPU environment YAML for vLLM serving:

```yaml
name: vllm-serving-env
channels:
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
      - azureml-inference-server-http==1.3.2
      - vllm==0.13.0
      - torch==2.9.0
      - transformers==4.57.3
      - sentencepiece
      - protobuf
```

A working CPU environment YAML for an orchestrator that coordinates calls to GPU endpoints:

```yaml
name: orchestrator-env
channels:
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
      - azureml-inference-server-http==1.3.2
      - sentence-transformers==3.0.1
      - torch==2.2.2
      - scikit-learn==1.5.0
      - numpy==1.26.4
      - pydantic==2.7.4
      - pyyaml==6.0.1
```

> *Environment versions are immutable. Once Azure ML registers an environment at a specific version number, it caches the built Docker image. Changing the conda YAML and re-registering with the same version number silently uses the old cached build. You must bump the version number every time you change the YAML. This is not documented prominently and causes hours of confusion when a "fixed" environment produces the exact same error.*

### 18.2 The azureml-inference-server-http Dependency Trap

The `azureml-inference-server-http` package is required for managed endpoints — it wraps your `init()` and `run()` functions into the HTTP server that Azure calls. But it pins strict sub-dependency versions that silently dictate your entire dependency tree.

Version 1.3.2 requires `pydantic~=2.7.1`. Pinning `pydantic==2.8.0` in the same YAML fails the Docker build with a pip resolver conflict. Version 1.5.0 requires `pydantic~=2.11.0` and specific `opentelemetry` versions. Upgrading the inference server without matching its telemetry pins breaks the build.

> *Before pinning any package version in your environment YAML, run `pip show azureml-inference-server-http` to see what it requires. Match your pins to its constraints, not the other way around. This single package is the most common hidden cause of environment build failures.*

### 18.3 Environment Reference Formats

When referencing an environment in a deployment, format matters. For environments in the same workspace, use `"azureml:my-env:2"` or just `"my-env:2"`. Using the full ARM resource path (`azureml:/subscriptions/.../environments/my-env/versions/2`) for same-workspace environments causes cryptic failures. The full path is only needed for cross-workspace references.

### 18.4 The Scoring Script Contract

Azure ML calls `init()` once when the container starts and `run(raw_data)` for each request. The contract is simple but the failure modes are subtle.

```python
import os, json, logging

logger = logging.getLogger(__name__)

def init():
    global model
    model_dir = os.environ.get("AZUREML_MODEL_DIR")
    logger.info(f"Model dir contents: {os.listdir(model_dir)}")
    # Load your model from model_dir

def run(raw_data: str) -> str:
    data = json.loads(raw_data)
    # ... inference ...
    return json.dumps({"result": "..."})
```

Four rules that each cost days of debugging when broken:

**`run()` must return a string.** Returning a Python dict instead of `json.dumps(result)` causes silent failures — the endpoint appears healthy but returns empty or malformed responses. The Azure inference server expects a serialized string and does not auto-convert.

**`AZUREML_MODEL_DIR` has a subdirectory.** When Azure downloads your registered model, it places the files inside a subdirectory of the model dir. Hardcoding the path without listing the directory first fails. Always discover the actual path at runtime:

```python
model_root = os.environ.get("AZUREML_MODEL_DIR")
subdirs = [d for d in os.listdir(model_root)
           if os.path.isdir(os.path.join(model_root, d))]
model_path = os.path.join(model_root, subdirs[0])
```

**Verify every import against your pinned version.** A class that exists in version 0.15 of a library may not exist in 0.13. If your environment pins `vllm==0.13.0` but your scoring script imports a class from 0.15, the container crashes during `init()` with an ImportError you cannot see until the container manages to start. Run `pip show <package>` on your compute instance and verify the actual API surface before writing any imports.

**Log everything in `init()`.** If `init()` fails, the container crashes silently and the deployment stays in "Creating" state indefinitely. Print the model directory listing, library versions, GPU device count, and available VRAM. This diagnostic output is the only thing that tells you what went wrong.

### 18.5 vLLM Scoring Script Pattern

For serving LLMs via vLLM inside Azure ML with an OpenAI-compatible interface:

```python
import os, json, logging
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def init():
    global llm, tokenizer
    model_dir = os.environ.get("AZUREML_MODEL_DIR")
    subdirs = [d for d in os.listdir(model_dir)
               if os.path.isdir(os.path.join(model_dir, d))]
    model_path = os.path.join(model_dir, subdirs[0]) if subdirs else model_dir

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="auto",              # bf16 on A100, fp16 on T4
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    logger.info("vLLM engine loaded")

def run(raw_data: str) -> str:
    data = json.loads(raw_data)
    messages = data.get("messages", [])
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    params = SamplingParams(
        temperature=data.get("temperature", 0.3),
        max_tokens=data.get("max_tokens", 2048),
        top_p=data.get("top_p", 0.95),
    )
    outputs = llm.generate([prompt], params)
    text = outputs[0].outputs[0].text
    return json.dumps({
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens": len(outputs[0].prompt_token_ids),
            "completion_tokens": len(outputs[0].outputs[0].token_ids),
        },
    })
```

The `dtype="auto"` setting is important — vLLM selects bf16 on A100 (which natively supports it) and fp16 on T4 (which does not). Forcing bf16 on a T4 causes a silent performance collapse because the hardware emulates it in software.

### 18.6 NER / Classification Scoring Script Pattern

For smaller models (token classification, span extraction, entity linking) that fit on a T4:

```python
import os, json

def init():
    global model
    model_root = os.environ.get("AZUREML_MODEL_DIR")
    subdirs = [d for d in os.listdir(model_root)
               if os.path.isdir(os.path.join(model_root, d))]
    model_path = os.path.join(model_root, subdirs[0])
    # model = YourNERModel.from_pretrained(model_path)
    # model = model.to("cuda")

def run(raw_data: str) -> str:
    data = json.loads(raw_data)
    text = data["text"]
    labels = data.get("labels", ["PERSON", "ORG", "CONDITION"])
    threshold = data.get("threshold", 0.5)
    entities = model.predict_entities(text, labels)
    filtered = [e for e in entities if e["score"] >= threshold]
    return json.dumps({"entities": filtered})
```

---

## 19. GPU SKU Selection and Model Sizing

Choosing the right GPU SKU is a sizing problem, not a feature comparison. The question is always: does the model fit in VRAM with enough headroom for the KV cache and intermediate activations?

### 19.1 Available SKUs

| SKU | GPU | VRAM | Approx. Cost | Best For |
| --- | --- | --- | --- | --- |
| Standard_NC4as_T4_v3 | T4 | 16 GB | ~$0.53/hr | NER, classifiers, models ≤7B |
| Standard_NC12s_v3 | V100 | 12 GB | ~$0.90/hr | Mid-size models, fp16 |
| Standard_NC24ads_A100_v4 | A100 | 80 GB | ~$3.67/hr | 20B+ params, vLLM serving |
| Standard_NC40ads_H100_v5 | H100 NVL | 94 GB | Higher | Faster A100 alternative |
| Standard_NV36ads_A10_v5 | A10 | 24 GB | Mid-range | Fallback when A100 is unavailable |

### 19.2 Model Size to GPU Mapping

A model in fp16 needs approximately **2× its parameter count in GB of VRAM**. Each parameter is 2 bytes. For fp32, multiply by 4. For 4-bit quantization (AWQ, GPTQ), divide the fp16 requirement by 4.

| Model Size | FP16 VRAM | 4-bit VRAM | Minimum SKU |
| --- | --- | --- | --- |
| 7B parameters | ~14 GB | ~3.5 GB | T4 (tight) or V100 |
| 14B parameters | ~28 GB | ~7 GB | A10 or A100 |
| 20B parameters | ~40 GB | ~10 GB | A100 |
| 70B parameters | ~140 GB | ~35 GB | A100 80GB (quantized only) |

These numbers are for weights alone. The KV cache, intermediate activations, and vLLM's memory management overhead add 10–20% on top. Multiply the weight estimate by 1.15 to get a practical minimum, then check that it sits below 85% of total VRAM to leave room for CUDA memory fragmentation.

> *We spent seven days debugging an endpoint that kept crashing on startup. The container would start, `init()` would begin loading the model, VRAM would fill, CUDA would throw out-of-memory, the container would crash, Azure would restart it, and the cycle would repeat. The deployment never left "Creating" state and logs were inaccessible because the container never lived long enough to produce them. The root cause was a 14.5 GB model (fp16) on a 12 GB GPU. The fix was using a larger SKU. The debugging cost far exceeded the price difference between T4 and A10.*

### 19.3 GPU Fallback Chains

Large GPU SKUs (A100, H100) are frequently out of capacity, especially in popular regions. A production deployment script should try SKUs in preference order and catch capacity errors to fall through:

```yaml
gpu_sku_fallback:
  - instance_type: "Standard_NC24ads_A100_v4"
    max_model_len: 8192
    gpu_memory_utilization: 0.90
  - instance_type: "Standard_NC40ads_H100_v5"
    max_model_len: 8192
    gpu_memory_utilization: 0.90
  - instance_type: "Standard_NV36ads_A10_v5"
    max_model_len: 4096     # degraded context window
    gpu_memory_utilization: 0.85
```

Each tier carries its own vLLM configuration to account for different VRAM budgets. The A10 fallback cuts context length in half — this is acceptable for many workloads and much better than failing to deploy entirely.

GPU quota is regional. Check your quota before deploying: some regions have zero available A100/H100 quota. Requesting an increase through the Azure portal takes 1–3 business days. Discover this during planning, not at deployment time.

---

## 20. Deployment Configuration

### 20.1 The Full Deployment Script

```python
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
    OnlineRequestSettings,
    ProbeSettings,
)

# 1. Create endpoint (the stable URL)
endpoint = ManagedOnlineEndpoint(
    name="my-llm-endpoint",
    auth_mode="key",
)
endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# 2. Create deployment (the running container)
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="my-llm-endpoint",
    model="azureml:my-model:1",
    environment="azureml:vllm-serving-env:2",
    code_configuration=CodeConfiguration(
        code="./scoring_scripts",
        scoring_script="score.py",
    ),
    instance_type="Standard_NC24ads_A100_v4",
    instance_count=1,
    request_settings=OnlineRequestSettings(
        request_timeout_ms=300000,       # 5 min per request
        max_concurrent_requests_per_instance=1,
        max_queue_wait_ms=300000,
    ),
    liveness_probe=ProbeSettings(
        initial_delay=600,    # 10 min before first health check
        timeout=30,
        period=100,
        failure_threshold=30,
    ),
    readiness_probe=ProbeSettings(
        initial_delay=600,
        timeout=30,
        period=30,
        failure_threshold=30,
    ),
)
deployment = ml_client.online_deployments.begin_create_or_update(deployment).result()

# 3. Route traffic
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
```

### 20.2 Why Probe Settings Matter

Large models take 5–10 minutes to load into GPU memory during `init()`. Azure ML uses liveness and readiness probes to check container health. If a probe fires before `init()` finishes loading the model, Azure concludes the container is unhealthy, kills it, and starts a new one. The new container also takes 5–10 minutes to load, gets killed, and the cycle repeats. The deployment stays in "Creating" state forever with no logs visible.

> *Set `initial_delay` to at least 600 seconds (10 minutes) for any deployment that loads a large model. For 70B+ models, use 900 seconds. This single setting prevents the most common invisible failure mode in GPU deployments. It is not documented prominently and there is no error message that tells you the probe is the problem — the deployment simply never transitions out of "Creating."*

### 20.3 The SDK Object Gotcha

The SDK requires `OnlineRequestSettings` and `ProbeSettings` to be proper SDK objects, not Python dicts. Passing a dict compiles without error and the deployment appears to succeed, but timeout behavior does not work as configured:

```python
# WRONG — dict compiles but fails silently
request_settings={"request_timeout_ms": 180000}

# CORRECT — use the SDK class
request_settings=OnlineRequestSettings(request_timeout_ms=180000)
```

### 20.4 Model Registration

Before deploying, register your model in the workspace:

```python
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

model = Model(
    name="my-model",
    version="1",
    path="/path/to/model/files",
    type=AssetTypes.CUSTOM_MODEL,
    description="Model description",
)
registered = ml_client.models.create_or_update(model)
```

For large models (40GB+), upload to Azure Blob Storage first, then register using the blob path. Direct upload from a compute instance is unreliable for files this size.

---

## 21. Blue-Green Deployments and Traffic Routing

Azure ML supports multiple deployments under a single endpoint. Each deployment runs independently with its own model version, environment, and SKU. Traffic is split by percentage. This enables canary rollouts, A/B testing, and zero-downtime model updates.

```python
# Deploy green alongside existing blue
green = ManagedOnlineDeployment(
    name="green",
    endpoint_name="my-llm-endpoint",
    # ... new model version, new environment ...
)
ml_client.online_deployments.begin_create_or_update(green).result()

# Canary: 10% to green
endpoint.traffic = {"blue": 90, "green": 10}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Full cutover after validation
endpoint.traffic = {"blue": 0, "green": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Remove old deployment to stop billing
ml_client.online_deployments.begin_delete(
    name="blue", endpoint_name="my-llm-endpoint"
).result()
```

You can bypass the traffic split entirely by adding the header `azureml-model-deployment: green` to a request, routing it to a specific deployment for testing before shifting any production traffic.

> *A newly created deployment receives 0% traffic by default. If you deploy and forget to set traffic, the endpoint returns errors for all requests because no deployment is receiving them. Always explicitly set traffic after creating a deployment.*

---

## 22. Multi-Endpoint Orchestrator Architecture

Complex ML pipelines that chain multiple models — summarization feeding into NER feeding into fact-checking — should not be deployed as a single monolithic endpoint. Deploy each model as its own endpoint and use a lightweight CPU orchestrator as the single client-facing entry point.

```
Client → Orchestrator (CPU endpoint)
              ├── LLM Summarizer   (GPU, A100)
              ├── NER Model A       (GPU, T4)
              ├── NER Model B       (GPU, T4)
              └── Entity Linker     (GPU or CPU)
```

The orchestrator receives the client request, fans out HTTP calls to sub-endpoints in parallel using `ThreadPoolExecutor`, collects and merges results, and returns a unified response. It runs on a CPU SKU (`Standard_F8s_v2`, ~$0.34/hr) because it does no inference — only coordination, lightweight embedding, and result formatting.

**Pass endpoint URLs and API keys in the request payload, not as environment variables.** This makes the system reconfigurable without redeploying the orchestrator. Swap a model endpoint, rotate an API key, or add a new NER model — all without touching the running orchestrator container.

Dictionary-based NER and entity linking can run on CPU if latency tolerance is 2–3 seconds per call instead of sub-second. For single-document inference, CPU is fine. For batch processing hundreds of documents, GPU is worth it.

### 22.1 Cost Example

| Component | SKU | Count | Cost/hr |
| --- | --- | --- | --- |
| Orchestrator | F8s_v2 (CPU) | 1 | ~$0.34 |
| LLM (20B params) | NC24ads A100 | 1 | ~$3.67 |
| NER Models | NC4as T4 | 3 | ~$1.59 |
| Entity Linkers | NC4as T4 | 3 | ~$1.59 |
| **Total** | | | **~$7.19** |

Adding a 70B model (AWQ 4-bit on A100) pushes the total to ~$11/hr. Evaluate whether the quality improvement justifies 3–4× the cost of the next tier down — often a distilled 20B model closes enough of the gap.

---

## 23. Debugging Failed Deployments

### 23.1 The "Creating" State Loop

The most frustrating failure mode: the deployment sits in "Creating" for 30+ minutes with no logs. The container has not started, so there is nothing to log. Diagnosis, in order of likelihood:

**Environment build timeout.** Check the build log in your workspace storage account. If the Docker build failed, no container was ever created.

**Probe restart loop.** If `initial_delay` is too short, the container loads for 5 minutes, gets killed by the liveness probe, restarts, loads again, gets killed again. Increase `initial_delay` to 600+ seconds.

**GPU quota exhausted.** The deployment is waiting for capacity that does not exist. Check quota with:

```python
from azure.mgmt.compute import ComputeManagementClient

compute_client = ComputeManagementClient(
    DefaultAzureCredential(), "your-subscription-id"
)
for u in compute_client.usage.list("your-region"):
    if any(k in u.name.localized_value.lower() for k in ["nc24", "a100", "gpu"]):
        print(f"{u.name.localized_value}: {u.current_value}/{u.limit}")
```

**`init()` crashing.** Import errors, CUDA OOM, model file not found. You cannot see these until the container runs long enough to produce logs. Once it does:

```python
logs = ml_client.online_deployments.get_logs(
    name="blue", endpoint_name="my-endpoint", lines=200
)
print(logs)
```

### 23.2 Common Error Patterns

| Error | Root Cause | Fix |
| --- | --- | --- |
| timeout waiting for Environment Image | Conda env too complex for Docker build | Simplify YAML, use pre-built base image |
| ImportError: cannot import name X | Library version mismatch | Verify import exists in pinned version |
| Stuck in Creating indefinitely | init() crash or probe restart loop | Increase initial_delay, add logging to init() |
| run() returns empty response | Returning dict instead of string | Always json.dumps() |
| pydantic version conflict | azureml-inference-server-http pins pydantic | Match pydantic to server requirements |
| CUDA out of memory | Model too large for GPU | Larger SKU or quantize the model |

### 23.3 The Runtime Installation Escape Hatch

When environment builds fail repeatedly — usually complex combinations of CUDA, PyTorch, and vLLM — there is a last-resort technique: use a minimal pre-built environment and install packages inside `init()` at runtime:

```python
import subprocess, sys, os

def init():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "transformers", "vllm", "--quiet"
    ])
    from vllm import LLM
    global llm
    llm = LLM(model=os.environ["AZUREML_MODEL_DIR"])
```

The first request takes 2–3 minutes while packages install. Subsequent requests use the cached installation. This bypasses the Docker build entirely. It is inelegant and should not be standard practice, but when nothing else works, it ships.

---

## 24. Deployment Lessons

These are patterns that emerged from months of production deployment work. None are in the official documentation.

### 24.1 Delete and Recreate; Never Update Failed Deployments

When a deployment fails, calling `begin_create_or_update` on the same deployment usually fails again with stale state — cached Docker layers, partially downloaded models, corrupted container state. The reliable pattern is `begin_delete()`, wait for completion, create fresh. This adds 5 minutes but prevents hours of chasing phantom errors from stale state. Treat failed deployments as disposable.

### 24.2 Register Environments Before Deploying

Do not combine environment registration and deployment into one step. Register the environment first with `ml_client.environments.create_or_update()`, wait for the Docker image to build, confirm success, then deploy. This separates two independent failure modes — environment build and container startup — so you diagnose each one clearly. Combined, a failure could be either, and the error messages do not distinguish.

### 24.3 The code_path Uploads Everything

When you specify `code="./scoring_scripts"` in CodeConfiguration, Azure uploads **every file** in that directory to the deployment container. If the directory contains model weights, datasets, notebooks, or anything large, the upload will be slow or fail. Keep the code directory to scoring scripts and small config files only. Model weights belong in registered Azure ML models, not in the code path.

### 24.4 Test Scoring Scripts Locally Before Deploying

Your Azure ML compute instance has the same GPU that your endpoint will use. Before deploying, copy the scoring script to the compute, import it, call `init()` to load the model, call `run()` with sample JSON. This catches 90% of issues — import errors, model path problems, VRAM limits, serialization bugs — for free, in seconds, with full stack traces. A failed deployment takes 10–30 minutes to surface the same error, if it surfaces at all.

### 24.5 Use Compute Instance Serving for Development

Managed endpoints are for production — when other systems need a stable URL. During development and benchmarking, run the model directly on your compute instance. A vLLM server on localhost or an Ollama instance gives you the same inference quality with zero deployment overhead, zero extra cost beyond the compute itself, and instant iteration. Deploy as a managed endpoint only when you need a stable URL for integration.

### 24.6 Pre-Deployment Checklist

| Check | Verification | Failure If Skipped |
| --- | --- | --- |
| Environment builds | Register separately, wait for Docker build | 30 min in Creating, then failure |
| Imports are valid | `pip show <pkg>` on compute instance | Container crash, no visible logs |
| `run()` returns `json.dumps()` | Test locally with sample input | Empty or malformed responses |
| Model fits GPU | Param count × 2 bytes ≤ VRAM × 0.85 | CUDA OOM, infinite restart loop |
| Probe initial_delay ≥ 600s | Check deployment config | Container killed before load, stuck in Creating |
| `azureml-inference-server-http` in env | Check conda YAML | No HTTP server, all requests fail |
| Traffic set after deployment | Explicitly set `endpoint.traffic` | 0% traffic, endpoint returns errors |

> *The most expensive lesson across all of this: the gap between "model works on a compute instance" and "model works as a managed endpoint" is almost entirely operational, not algorithmic. The model is the same. The math is the same. What changes is the packaging, the probes, the environment builds, the traffic routing, and the debugging visibility. Mastering this operational layer is what separates an ML engineer who builds prototypes from one who runs production systems.*

---


## 18. Production GPU Deployment on Azure ML (SDK v2)

Everything up to this point assumes you have a model and want to make it better, faster, or smaller. This section assumes you have a model and need to put it behind a URL that other systems can call reliably. The gap between "it works on my machine" and "it works as a managed endpoint" is where most ML engineering time actually goes.

Azure ML SDK v2 organizes deployment around five objects: **MLClient** (authentication), **Endpoint** (the stable URL), **Deployment** (the running container behind that URL), **Environment** (the Docker image plus packages), and **CodeConfiguration** (the scoring script). Understanding the boundaries between these objects prevents most first-time deployment failures.

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="your-sub-id",
    resource_group_name="your-rg",
    workspace_name="your-workspace",
)
```

`DefaultAzureCredential` works automatically from Azure ML compute instances. Use `AzureCliCredential` after `az login` when running from a local machine.

### 18.1 The Environment Build Problem

Azure ML builds a Docker image from your conda YAML plus a base image. Complex environments — anything combining PyTorch, CUDA, transformers, and vLLM — routinely time out during the Docker build step. This is the single biggest source of failed deployments, and it happens before your code even runs.

Three principles that prevent most build failures: keep conda YAMLs minimal with exact version pins and don't mix conda channels with pip for the same library. Use Microsoft's CUDA base images (`mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:latest`) instead of building CUDA from scratch. Register the environment separately before deploying — this isolates the Docker build from the model download and container startup, so you know immediately which step failed.

When a build times out, Azure throws `ResourceOperationFailure` with "timeout while waiting for Environment Image." The actual build log lives in your workspace storage account under `aml-environment-image-build/{env-name}/{version}/`. The error message from Azure itself is never specific enough to diagnose the problem — always check the build log directly.

A working GPU environment YAML for vLLM serving:

```yaml
name: vllm-serving-env
channels:
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
      - azureml-inference-server-http==1.3.2
      - vllm==0.13.0
      - torch==2.9.0
      - transformers==4.57.3
      - sentencepiece
      - protobuf
```

A working CPU environment YAML for an orchestrator that coordinates calls to GPU endpoints:

```yaml
name: orchestrator-env
channels:
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
      - azureml-inference-server-http==1.3.2
      - sentence-transformers==3.0.1
      - torch==2.2.2
      - scikit-learn==1.5.0
      - numpy==1.26.4
      - pydantic==2.7.4
      - pyyaml==6.0.1
```

> *Environment versions are immutable. Once Azure ML registers an environment at a specific version number, it caches the built Docker image. Changing the conda YAML and re-registering with the same version number silently uses the old cached build. You must bump the version number every time you change the YAML. This is not documented prominently and causes hours of confusion when a "fixed" environment produces the exact same error.*

### 18.2 The azureml-inference-server-http Dependency Trap

The `azureml-inference-server-http` package is required for managed endpoints — it wraps your `init()` and `run()` functions into the HTTP server that Azure calls. But it pins strict sub-dependency versions that silently dictate your entire dependency tree.

Version 1.3.2 requires `pydantic~=2.7.1`. Pinning `pydantic==2.8.0` in the same YAML fails the Docker build with a pip resolver conflict. Version 1.5.0 requires `pydantic~=2.11.0` and specific `opentelemetry` versions. Upgrading the inference server without matching its telemetry pins breaks the build.

> *Before pinning any package version in your environment YAML, run `pip show azureml-inference-server-http` to see what it requires. Match your pins to its constraints, not the other way around. This single package is the most common hidden cause of environment build failures.*

### 18.3 Environment Reference Formats

When referencing an environment in a deployment, format matters. For environments in the same workspace, use `"azureml:my-env:2"` or just `"my-env:2"`. Using the full ARM resource path (`azureml:/subscriptions/.../environments/my-env/versions/2`) for same-workspace environments causes cryptic failures. The full path is only needed for cross-workspace references.

### 18.4 The Scoring Script Contract

Azure ML calls `init()` once when the container starts and `run(raw_data)` for each request. The contract is simple but the failure modes are subtle.

```python
import os, json, logging

logger = logging.getLogger(__name__)

def init():
    global model
    model_dir = os.environ.get("AZUREML_MODEL_DIR")
    logger.info(f"Model dir contents: {os.listdir(model_dir)}")
    # Load your model from model_dir

def run(raw_data: str) -> str:
    data = json.loads(raw_data)
    # ... inference ...
    return json.dumps({"result": "..."})
```

Four rules that each cost days of debugging when broken:

**`run()` must return a string.** Returning a Python dict instead of `json.dumps(result)` causes silent failures — the endpoint appears healthy but returns empty or malformed responses. The Azure inference server expects a serialized string and does not auto-convert.

**`AZUREML_MODEL_DIR` has a subdirectory.** When Azure downloads your registered model, it places the files inside a subdirectory of the model dir. Hardcoding the path without listing the directory first fails. Always discover the actual path at runtime:

```python
model_root = os.environ.get("AZUREML_MODEL_DIR")
subdirs = [d for d in os.listdir(model_root)
           if os.path.isdir(os.path.join(model_root, d))]
model_path = os.path.join(model_root, subdirs[0])
```

**Verify every import against your pinned version.** A class that exists in version 0.15 of a library may not exist in 0.13. If your environment pins `vllm==0.13.0` but your scoring script imports a class from 0.15, the container crashes during `init()` with an ImportError you cannot see until the container manages to start. Run `pip show <package>` on your compute instance and verify the actual API surface before writing any imports.

**Log everything in `init()`.** If `init()` fails, the container crashes silently and the deployment stays in "Creating" state indefinitely. Print the model directory listing, library versions, GPU device count, and available VRAM. This diagnostic output is the only thing that tells you what went wrong.

### 18.5 vLLM Scoring Script Pattern

For serving LLMs via vLLM inside Azure ML with an OpenAI-compatible interface:

```python
import os, json, logging
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def init():
    global llm, tokenizer
    model_dir = os.environ.get("AZUREML_MODEL_DIR")
    subdirs = [d for d in os.listdir(model_dir)
               if os.path.isdir(os.path.join(model_dir, d))]
    model_path = os.path.join(model_dir, subdirs[0]) if subdirs else model_dir

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="auto",              # bf16 on A100, fp16 on T4
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    logger.info("vLLM engine loaded")

def run(raw_data: str) -> str:
    data = json.loads(raw_data)
    messages = data.get("messages", [])
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    params = SamplingParams(
        temperature=data.get("temperature", 0.3),
        max_tokens=data.get("max_tokens", 2048),
        top_p=data.get("top_p", 0.95),
    )
    outputs = llm.generate([prompt], params)
    text = outputs[0].outputs[0].text
    return json.dumps({
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens": len(outputs[0].prompt_token_ids),
            "completion_tokens": len(outputs[0].outputs[0].token_ids),
        },
    })
```

The `dtype="auto"` setting is important — vLLM selects bf16 on A100 (which natively supports it) and fp16 on T4 (which does not). Forcing bf16 on a T4 causes a silent performance collapse because the hardware emulates it in software.

### 18.6 NER / Classification Scoring Script Pattern

For smaller models (token classification, span extraction, entity linking) that fit on a T4:

```python
import os, json

def init():
    global model
    model_root = os.environ.get("AZUREML_MODEL_DIR")
    subdirs = [d for d in os.listdir(model_root)
               if os.path.isdir(os.path.join(model_root, d))]
    model_path = os.path.join(model_root, subdirs[0])
    # model = YourNERModel.from_pretrained(model_path)
    # model = model.to("cuda")

def run(raw_data: str) -> str:
    data = json.loads(raw_data)
    text = data["text"]
    labels = data.get("labels", ["PERSON", "ORG", "CONDITION"])
    threshold = data.get("threshold", 0.5)
    entities = model.predict_entities(text, labels)
    filtered = [e for e in entities if e["score"] >= threshold]
    return json.dumps({"entities": filtered})
```

---

## 19. GPU SKU Selection and Model Sizing

Choosing the right GPU SKU is a sizing problem, not a feature comparison. The question is always: does the model fit in VRAM with enough headroom for the KV cache and intermediate activations?

### 19.1 Available SKUs

| SKU | GPU | VRAM | Approx. Cost | Best For |
| --- | --- | --- | --- | --- |
| Standard_NC4as_T4_v3 | T4 | 16 GB | ~$0.53/hr | NER, classifiers, models ≤7B |
| Standard_NC12s_v3 | V100 | 12 GB | ~$0.90/hr | Mid-size models, fp16 |
| Standard_NC24ads_A100_v4 | A100 | 80 GB | ~$3.67/hr | 20B+ params, vLLM serving |
| Standard_NC40ads_H100_v5 | H100 NVL | 94 GB | Higher | Faster A100 alternative |
| Standard_NV36ads_A10_v5 | A10 | 24 GB | Mid-range | Fallback when A100 is unavailable |

### 19.2 Model Size to GPU Mapping

A model in fp16 needs approximately **2× its parameter count in GB of VRAM**. Each parameter is 2 bytes. For fp32, multiply by 4. For 4-bit quantization (AWQ, GPTQ), divide the fp16 requirement by 4.

| Model Size | FP16 VRAM | 4-bit VRAM | Minimum SKU |
| --- | --- | --- | --- |
| 7B parameters | ~14 GB | ~3.5 GB | T4 (tight) or V100 |
| 14B parameters | ~28 GB | ~7 GB | A10 or A100 |
| 20B parameters | ~40 GB | ~10 GB | A100 |
| 70B parameters | ~140 GB | ~35 GB | A100 80GB (quantized only) |

These numbers are for weights alone. The KV cache, intermediate activations, and vLLM's memory management overhead add 10–20% on top. Multiply the weight estimate by 1.15 to get a practical minimum, then check that it sits below 85% of total VRAM to leave room for CUDA memory fragmentation.

> *We spent seven days debugging an endpoint that kept crashing on startup. The container would start, `init()` would begin loading the model, VRAM would fill, CUDA would throw out-of-memory, the container would crash, Azure would restart it, and the cycle would repeat. The deployment never left "Creating" state and logs were inaccessible because the container never lived long enough to produce them. The root cause was a 14.5 GB model (fp16) on a 12 GB GPU. The fix was using a larger SKU. The debugging cost far exceeded the price difference between T4 and A10.*

### 19.3 GPU Fallback Chains

Large GPU SKUs (A100, H100) are frequently out of capacity, especially in popular regions. A production deployment script should try SKUs in preference order and catch capacity errors to fall through:

```yaml
gpu_sku_fallback:
  - instance_type: "Standard_NC24ads_A100_v4"
    max_model_len: 8192
    gpu_memory_utilization: 0.90
  - instance_type: "Standard_NC40ads_H100_v5"
    max_model_len: 8192
    gpu_memory_utilization: 0.90
  - instance_type: "Standard_NV36ads_A10_v5"
    max_model_len: 4096     # degraded context window
    gpu_memory_utilization: 0.85
```

Each tier carries its own vLLM configuration to account for different VRAM budgets. The A10 fallback cuts context length in half — this is acceptable for many workloads and much better than failing to deploy entirely.

GPU quota is regional. Check your quota before deploying: some regions have zero available A100/H100 quota. Requesting an increase through the Azure portal takes 1–3 business days. Discover this during planning, not at deployment time.

---

## 20. Deployment Configuration

### 20.1 The Full Deployment Script

```python
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
    OnlineRequestSettings,
    ProbeSettings,
)

# 1. Create endpoint (the stable URL)
endpoint = ManagedOnlineEndpoint(
    name="my-llm-endpoint",
    auth_mode="key",
)
endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# 2. Create deployment (the running container)
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="my-llm-endpoint",
    model="azureml:my-model:1",
    environment="azureml:vllm-serving-env:2",
    code_configuration=CodeConfiguration(
        code="./scoring_scripts",
        scoring_script="score.py",
    ),
    instance_type="Standard_NC24ads_A100_v4",
    instance_count=1,
    request_settings=OnlineRequestSettings(
        request_timeout_ms=300000,       # 5 min per request
        max_concurrent_requests_per_instance=1,
        max_queue_wait_ms=300000,
    ),
    liveness_probe=ProbeSettings(
        initial_delay=600,    # 10 min before first health check
        timeout=30,
        period=100,
        failure_threshold=30,
    ),
    readiness_probe=ProbeSettings(
        initial_delay=600,
        timeout=30,
        period=30,
        failure_threshold=30,
    ),
)
deployment = ml_client.online_deployments.begin_create_or_update(deployment).result()

# 3. Route traffic
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
```

### 20.2 Why Probe Settings Matter

Large models take 5–10 minutes to load into GPU memory during `init()`. Azure ML uses liveness and readiness probes to check container health. If a probe fires before `init()` finishes loading the model, Azure concludes the container is unhealthy, kills it, and starts a new one. The new container also takes 5–10 minutes to load, gets killed, and the cycle repeats. The deployment stays in "Creating" state forever with no logs visible.

> *Set `initial_delay` to at least 600 seconds (10 minutes) for any deployment that loads a large model. For 70B+ models, use 900 seconds. This single setting prevents the most common invisible failure mode in GPU deployments. It is not documented prominently and there is no error message that tells you the probe is the problem — the deployment simply never transitions out of "Creating."*

### 20.3 The SDK Object Gotcha

The SDK requires `OnlineRequestSettings` and `ProbeSettings` to be proper SDK objects, not Python dicts. Passing a dict compiles without error and the deployment appears to succeed, but timeout behavior does not work as configured:

```python
# WRONG — dict compiles but fails silently
request_settings={"request_timeout_ms": 180000}

# CORRECT — use the SDK class
request_settings=OnlineRequestSettings(request_timeout_ms=180000)
```

### 20.4 Model Registration

Before deploying, register your model in the workspace:

```python
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

model = Model(
    name="my-model",
    version="1",
    path="/path/to/model/files",
    type=AssetTypes.CUSTOM_MODEL,
    description="Model description",
)
registered = ml_client.models.create_or_update(model)
```

For large models (40GB+), upload to Azure Blob Storage first, then register using the blob path. Direct upload from a compute instance is unreliable for files this size.

---

## 21. Blue-Green Deployments and Traffic Routing

Azure ML supports multiple deployments under a single endpoint. Each deployment runs independently with its own model version, environment, and SKU. Traffic is split by percentage. This enables canary rollouts, A/B testing, and zero-downtime model updates.

```python
# Deploy green alongside existing blue
green = ManagedOnlineDeployment(
    name="green",
    endpoint_name="my-llm-endpoint",
    # ... new model version, new environment ...
)
ml_client.online_deployments.begin_create_or_update(green).result()

# Canary: 10% to green
endpoint.traffic = {"blue": 90, "green": 10}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Full cutover after validation
endpoint.traffic = {"blue": 0, "green": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Remove old deployment to stop billing
ml_client.online_deployments.begin_delete(
    name="blue", endpoint_name="my-llm-endpoint"
).result()
```

You can bypass the traffic split entirely by adding the header `azureml-model-deployment: green` to a request, routing it to a specific deployment for testing before shifting any production traffic.

> *A newly created deployment receives 0% traffic by default. If you deploy and forget to set traffic, the endpoint returns errors for all requests because no deployment is receiving them. Always explicitly set traffic after creating a deployment.*

---

## 22. Multi-Endpoint Orchestrator Architecture

Complex ML pipelines that chain multiple models — summarization feeding into NER feeding into fact-checking — should not be deployed as a single monolithic endpoint. Deploy each model as its own endpoint and use a lightweight CPU orchestrator as the single client-facing entry point.

```
Client → Orchestrator (CPU endpoint)
              ├── LLM Summarizer   (GPU, A100)
              ├── NER Model A       (GPU, T4)
              ├── NER Model B       (GPU, T4)
              └── Entity Linker     (GPU or CPU)
```

The orchestrator receives the client request, fans out HTTP calls to sub-endpoints in parallel using `ThreadPoolExecutor`, collects and merges results, and returns a unified response. It runs on a CPU SKU (`Standard_F8s_v2`, ~$0.34/hr) because it does no inference — only coordination, lightweight embedding, and result formatting.

**Pass endpoint URLs and API keys in the request payload, not as environment variables.** This makes the system reconfigurable without redeploying the orchestrator. Swap a model endpoint, rotate an API key, or add a new NER model — all without touching the running orchestrator container.

Dictionary-based NER and entity linking can run on CPU if latency tolerance is 2–3 seconds per call instead of sub-second. For single-document inference, CPU is fine. For batch processing hundreds of documents, GPU is worth it.

### 22.1 Cost Example

| Component | SKU | Count | Cost/hr |
| --- | --- | --- | --- |
| Orchestrator | F8s_v2 (CPU) | 1 | ~$0.34 |
| LLM (20B params) | NC24ads A100 | 1 | ~$3.67 |
| NER Models | NC4as T4 | 3 | ~$1.59 |
| Entity Linkers | NC4as T4 | 3 | ~$1.59 |
| **Total** | | | **~$7.19** |

Adding a 70B model (AWQ 4-bit on A100) pushes the total to ~$11/hr. Evaluate whether the quality improvement justifies 3–4× the cost of the next tier down — often a distilled 20B model closes enough of the gap.

---

## 23. Debugging Failed Deployments

### 23.1 The "Creating" State Loop

The most frustrating failure mode: the deployment sits in "Creating" for 30+ minutes with no logs. The container has not started, so there is nothing to log. Diagnosis, in order of likelihood:

**Environment build timeout.** Check the build log in your workspace storage account. If the Docker build failed, no container was ever created.

**Probe restart loop.** If `initial_delay` is too short, the container loads for 5 minutes, gets killed by the liveness probe, restarts, loads again, gets killed again. Increase `initial_delay` to 600+ seconds.

**GPU quota exhausted.** The deployment is waiting for capacity that does not exist. Check quota with:

```python
from azure.mgmt.compute import ComputeManagementClient

compute_client = ComputeManagementClient(
    DefaultAzureCredential(), "your-subscription-id"
)
for u in compute_client.usage.list("your-region"):
    if any(k in u.name.localized_value.lower() for k in ["nc24", "a100", "gpu"]):
        print(f"{u.name.localized_value}: {u.current_value}/{u.limit}")
```

**`init()` crashing.** Import errors, CUDA OOM, model file not found. You cannot see these until the container runs long enough to produce logs. Once it does:

```python
logs = ml_client.online_deployments.get_logs(
    name="blue", endpoint_name="my-endpoint", lines=200
)
print(logs)
```

### 23.2 Common Error Patterns

| Error | Root Cause | Fix |
| --- | --- | --- |
| timeout waiting for Environment Image | Conda env too complex for Docker build | Simplify YAML, use pre-built base image |
| ImportError: cannot import name X | Library version mismatch | Verify import exists in pinned version |
| Stuck in Creating indefinitely | init() crash or probe restart loop | Increase initial_delay, add logging to init() |
| run() returns empty response | Returning dict instead of string | Always json.dumps() |
| pydantic version conflict | azureml-inference-server-http pins pydantic | Match pydantic to server requirements |
| CUDA out of memory | Model too large for GPU | Larger SKU or quantize the model |

### 23.3 The Runtime Installation Escape Hatch

When environment builds fail repeatedly — usually complex combinations of CUDA, PyTorch, and vLLM — there is a last-resort technique: use a minimal pre-built environment and install packages inside `init()` at runtime:

```python
import subprocess, sys, os

def init():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "transformers", "vllm", "--quiet"
    ])
    from vllm import LLM
    global llm
    llm = LLM(model=os.environ["AZUREML_MODEL_DIR"])
```

The first request takes 2–3 minutes while packages install. Subsequent requests use the cached installation. This bypasses the Docker build entirely. It is inelegant and should not be standard practice, but when nothing else works, it ships.

---

## 24. Deployment Lessons

These are patterns that emerged from months of production deployment work. None are in the official documentation.

### 24.1 Delete and Recreate; Never Update Failed Deployments

When a deployment fails, calling `begin_create_or_update` on the same deployment usually fails again with stale state — cached Docker layers, partially downloaded models, corrupted container state. The reliable pattern is `begin_delete()`, wait for completion, create fresh. This adds 5 minutes but prevents hours of chasing phantom errors from stale state. Treat failed deployments as disposable.

### 24.2 Register Environments Before Deploying

Do not combine environment registration and deployment into one step. Register the environment first with `ml_client.environments.create_or_update()`, wait for the Docker image to build, confirm success, then deploy. This separates two independent failure modes — environment build and container startup — so you diagnose each one clearly. Combined, a failure could be either, and the error messages do not distinguish.

### 24.3 The code_path Uploads Everything

When you specify `code="./scoring_scripts"` in CodeConfiguration, Azure uploads **every file** in that directory to the deployment container. If the directory contains model weights, datasets, notebooks, or anything large, the upload will be slow or fail. Keep the code directory to scoring scripts and small config files only. Model weights belong in registered Azure ML models, not in the code path.

### 24.4 Test Scoring Scripts Locally Before Deploying

Your Azure ML compute instance has the same GPU that your endpoint will use. Before deploying, copy the scoring script to the compute, import it, call `init()` to load the model, call `run()` with sample JSON. This catches 90% of issues — import errors, model path problems, VRAM limits, serialization bugs — for free, in seconds, with full stack traces. A failed deployment takes 10–30 minutes to surface the same error, if it surfaces at all.

### 24.5 Use Compute Instance Serving for Development

Managed endpoints are for production — when other systems need a stable URL. During development and benchmarking, run the model directly on your compute instance. A vLLM server on localhost or an Ollama instance gives you the same inference quality with zero deployment overhead, zero extra cost beyond the compute itself, and instant iteration. Deploy as a managed endpoint only when you need a stable URL for integration.

### 24.6 Pre-Deployment Checklist

| Check | Verification | Failure If Skipped |
| --- | --- | --- |
| Environment builds | Register separately, wait for Docker build | 30 min in Creating, then failure |
| Imports are valid | `pip show <pkg>` on compute instance | Container crash, no visible logs |
| `run()` returns `json.dumps()` | Test locally with sample input | Empty or malformed responses |
| Model fits GPU | Param count × 2 bytes ≤ VRAM × 0.85 | CUDA OOM, infinite restart loop |
| Probe initial_delay ≥ 600s | Check deployment config | Container killed before load, stuck in Creating |
| `azureml-inference-server-http` in env | Check conda YAML | No HTTP server, all requests fail |
| Traffic set after deployment | Explicitly set `endpoint.traffic` | 0% traffic, endpoint returns errors |

> *The most expensive lesson across all of this: the gap between "model works on a compute instance" and "model works as a managed endpoint" is almost entirely operational, not algorithmic. The model is the same. The math is the same. What changes is the packaging, the probes, the environment builds, the traffic routing, and the debugging visibility. Mastering this operational layer is what separates an ML engineer who builds prototypes from one who runs production systems.*

---


## 18. Production GPU Deployment on Azure ML (SDK v2)

Everything up to this point assumes you have a model and want to make it better, faster, or smaller. This section assumes you have a model and need to put it behind a URL that other systems can call reliably. The gap between "it works on my machine" and "it works as a managed endpoint" is where most ML engineering time actually goes.

Azure ML SDK v2 organizes deployment around five objects: **MLClient** (authentication), **Endpoint** (the stable URL), **Deployment** (the running container behind that URL), **Environment** (the Docker image plus packages), and **CodeConfiguration** (the scoring script). Understanding the boundaries between these objects prevents most first-time deployment failures.

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="your-sub-id",
    resource_group_name="your-rg",
    workspace_name="your-workspace",
)
```

`DefaultAzureCredential` works automatically from Azure ML compute instances. Use `AzureCliCredential` after `az login` when running from a local machine.

### 18.1 The Environment Build Problem

Azure ML builds a Docker image from your conda YAML plus a base image. Complex environments — anything combining PyTorch, CUDA, transformers, and vLLM — routinely time out during the Docker build step. This is the single biggest source of failed deployments, and it happens before your code even runs.

Three principles that prevent most build failures: keep conda YAMLs minimal with exact version pins and don't mix conda channels with pip for the same library. Use Microsoft's CUDA base images (`mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:latest`) instead of building CUDA from scratch. Register the environment separately before deploying — this isolates the Docker build from the model download and container startup, so you know immediately which step failed.

When a build times out, Azure throws `ResourceOperationFailure` with "timeout while waiting for Environment Image." The actual build log lives in your workspace storage account under `aml-environment-image-build/{env-name}/{version}/`. The error message from Azure itself is never specific enough to diagnose the problem — always check the build log directly.

A working GPU environment YAML for vLLM serving:

```yaml
name: vllm-serving-env
channels:
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
      - azureml-inference-server-http==1.3.2
      - vllm==0.13.0
      - torch==2.9.0
      - transformers==4.57.3
      - sentencepiece
      - protobuf
```

A working CPU environment YAML for an orchestrator that coordinates calls to GPU endpoints:

```yaml
name: orchestrator-env
channels:
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
      - azureml-inference-server-http==1.3.2
      - sentence-transformers==3.0.1
      - torch==2.2.2
      - scikit-learn==1.5.0
      - numpy==1.26.4
      - pydantic==2.7.4
      - pyyaml==6.0.1
```

> *Environment versions are immutable. Once Azure ML registers an environment at a specific version number, it caches the built Docker image. Changing the conda YAML and re-registering with the same version number silently uses the old cached build. You must bump the version number every time you change the YAML. This is not documented prominently and causes hours of confusion when a "fixed" environment produces the exact same error.*

### 18.2 The azureml-inference-server-http Dependency Trap

The `azureml-inference-server-http` package is required for managed endpoints — it wraps your `init()` and `run()` functions into the HTTP server that Azure calls. But it pins strict sub-dependency versions that silently dictate your entire dependency tree.

Version 1.3.2 requires `pydantic~=2.7.1`. Pinning `pydantic==2.8.0` in the same YAML fails the Docker build with a pip resolver conflict. Version 1.5.0 requires `pydantic~=2.11.0` and specific `opentelemetry` versions. Upgrading the inference server without matching its telemetry pins breaks the build.

> *Before pinning any package version in your environment YAML, run `pip show azureml-inference-server-http` to see what it requires. Match your pins to its constraints, not the other way around. This single package is the most common hidden cause of environment build failures.*

### 18.3 Environment Reference Formats

When referencing an environment in a deployment, format matters. For environments in the same workspace, use `"azureml:my-env:2"` or just `"my-env:2"`. Using the full ARM resource path (`azureml:/subscriptions/.../environments/my-env/versions/2`) for same-workspace environments causes cryptic failures. The full path is only needed for cross-workspace references.

### 18.4 The Scoring Script Contract

Azure ML calls `init()` once when the container starts and `run(raw_data)` for each request. The contract is simple but the failure modes are subtle.

```python
import os, json, logging

logger = logging.getLogger(__name__)

def init():
    global model
    model_dir = os.environ.get("AZUREML_MODEL_DIR")
    logger.info(f"Model dir contents: {os.listdir(model_dir)}")
    # Load your model from model_dir

def run(raw_data: str) -> str:
    data = json.loads(raw_data)
    # ... inference ...
    return json.dumps({"result": "..."})
```

Four rules that each cost days of debugging when broken:

**`run()` must return a string.** Returning a Python dict instead of `json.dumps(result)` causes silent failures — the endpoint appears healthy but returns empty or malformed responses. The Azure inference server expects a serialized string and does not auto-convert.

**`AZUREML_MODEL_DIR` has a subdirectory.** When Azure downloads your registered model, it places the files inside a subdirectory of the model dir. Hardcoding the path without listing the directory first fails. Always discover the actual path at runtime:

```python
model_root = os.environ.get("AZUREML_MODEL_DIR")
subdirs = [d for d in os.listdir(model_root)
           if os.path.isdir(os.path.join(model_root, d))]
model_path = os.path.join(model_root, subdirs[0])
```

**Verify every import against your pinned version.** A class that exists in version 0.15 of a library may not exist in 0.13. If your environment pins `vllm==0.13.0` but your scoring script imports a class from 0.15, the container crashes during `init()` with an ImportError you cannot see until the container manages to start. Run `pip show <package>` on your compute instance and verify the actual API surface before writing any imports.

**Log everything in `init()`.** If `init()` fails, the container crashes silently and the deployment stays in "Creating" state indefinitely. Print the model directory listing, library versions, GPU device count, and available VRAM. This diagnostic output is the only thing that tells you what went wrong.

### 18.5 vLLM Scoring Script Pattern

For serving LLMs via vLLM inside Azure ML with an OpenAI-compatible interface:

```python
import os, json, logging
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def init():
    global llm, tokenizer
    model_dir = os.environ.get("AZUREML_MODEL_DIR")
    subdirs = [d for d in os.listdir(model_dir)
               if os.path.isdir(os.path.join(model_dir, d))]
    model_path = os.path.join(model_dir, subdirs[0]) if subdirs else model_dir

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="auto",              # bf16 on A100, fp16 on T4
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    logger.info("vLLM engine loaded")

def run(raw_data: str) -> str:
    data = json.loads(raw_data)
    messages = data.get("messages", [])
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    params = SamplingParams(
        temperature=data.get("temperature", 0.3),
        max_tokens=data.get("max_tokens", 2048),
        top_p=data.get("top_p", 0.95),
    )
    outputs = llm.generate([prompt], params)
    text = outputs[0].outputs[0].text
    return json.dumps({
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens": len(outputs[0].prompt_token_ids),
            "completion_tokens": len(outputs[0].outputs[0].token_ids),
        },
    })
```

The `dtype="auto"` setting is important — vLLM selects bf16 on A100 (which natively supports it) and fp16 on T4 (which does not). Forcing bf16 on a T4 causes a silent performance collapse because the hardware emulates it in software.

### 18.6 NER / Classification Scoring Script Pattern

For smaller models (token classification, span extraction, entity linking) that fit on a T4:

```python
import os, json

def init():
    global model
    model_root = os.environ.get("AZUREML_MODEL_DIR")
    subdirs = [d for d in os.listdir(model_root)
               if os.path.isdir(os.path.join(model_root, d))]
    model_path = os.path.join(model_root, subdirs[0])
    # model = YourNERModel.from_pretrained(model_path)
    # model = model.to("cuda")

def run(raw_data: str) -> str:
    data = json.loads(raw_data)
    text = data["text"]
    labels = data.get("labels", ["PERSON", "ORG", "CONDITION"])
    threshold = data.get("threshold", 0.5)
    entities = model.predict_entities(text, labels)
    filtered = [e for e in entities if e["score"] >= threshold]
    return json.dumps({"entities": filtered})
```

---

## 19. GPU SKU Selection and Model Sizing

Choosing the right GPU SKU is a sizing problem, not a feature comparison. The question is always: does the model fit in VRAM with enough headroom for the KV cache and intermediate activations?

### 19.1 Available SKUs

| SKU | GPU | VRAM | Approx. Cost | Best For |
| --- | --- | --- | --- | --- |
| Standard_NC4as_T4_v3 | T4 | 16 GB | ~$0.53/hr | NER, classifiers, models ≤7B |
| Standard_NC12s_v3 | V100 | 12 GB | ~$0.90/hr | Mid-size models, fp16 |
| Standard_NC24ads_A100_v4 | A100 | 80 GB | ~$3.67/hr | 20B+ params, vLLM serving |
| Standard_NC40ads_H100_v5 | H100 NVL | 94 GB | Higher | Faster A100 alternative |
| Standard_NV36ads_A10_v5 | A10 | 24 GB | Mid-range | Fallback when A100 is unavailable |

### 19.2 Model Size to GPU Mapping

A model in fp16 needs approximately **2× its parameter count in GB of VRAM**. Each parameter is 2 bytes. For fp32, multiply by 4. For 4-bit quantization (AWQ, GPTQ), divide the fp16 requirement by 4.

| Model Size | FP16 VRAM | 4-bit VRAM | Minimum SKU |
| --- | --- | --- | --- |
| 7B parameters | ~14 GB | ~3.5 GB | T4 (tight) or V100 |
| 14B parameters | ~28 GB | ~7 GB | A10 or A100 |
| 20B parameters | ~40 GB | ~10 GB | A100 |
| 70B parameters | ~140 GB | ~35 GB | A100 80GB (quantized only) |

These numbers are for weights alone. The KV cache, intermediate activations, and vLLM's memory management overhead add 10–20% on top. Multiply the weight estimate by 1.15 to get a practical minimum, then check that it sits below 85% of total VRAM to leave room for CUDA memory fragmentation.

> *We spent seven days debugging an endpoint that kept crashing on startup. The container would start, `init()` would begin loading the model, VRAM would fill, CUDA would throw out-of-memory, the container would crash, Azure would restart it, and the cycle would repeat. The deployment never left "Creating" state and logs were inaccessible because the container never lived long enough to produce them. The root cause was a 14.5 GB model (fp16) on a 12 GB GPU. The fix was using a larger SKU. The debugging cost far exceeded the price difference between T4 and A10.*

### 19.3 GPU Fallback Chains

Large GPU SKUs (A100, H100) are frequently out of capacity, especially in popular regions. A production deployment script should try SKUs in preference order and catch capacity errors to fall through:

```yaml
gpu_sku_fallback:
  - instance_type: "Standard_NC24ads_A100_v4"
    max_model_len: 8192
    gpu_memory_utilization: 0.90
  - instance_type: "Standard_NC40ads_H100_v5"
    max_model_len: 8192
    gpu_memory_utilization: 0.90
  - instance_type: "Standard_NV36ads_A10_v5"
    max_model_len: 4096     # degraded context window
    gpu_memory_utilization: 0.85
```

Each tier carries its own vLLM configuration to account for different VRAM budgets. The A10 fallback cuts context length in half — this is acceptable for many workloads and much better than failing to deploy entirely.

GPU quota is regional. Check your quota before deploying: some regions have zero available A100/H100 quota. Requesting an increase through the Azure portal takes 1–3 business days. Discover this during planning, not at deployment time.

---

## 20. Deployment Configuration

### 20.1 The Full Deployment Script

```python
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
    OnlineRequestSettings,
    ProbeSettings,
)

# 1. Create endpoint (the stable URL)
endpoint = ManagedOnlineEndpoint(
    name="my-llm-endpoint",
    auth_mode="key",
)
endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# 2. Create deployment (the running container)
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="my-llm-endpoint",
    model="azureml:my-model:1",
    environment="azureml:vllm-serving-env:2",
    code_configuration=CodeConfiguration(
        code="./scoring_scripts",
        scoring_script="score.py",
    ),
    instance_type="Standard_NC24ads_A100_v4",
    instance_count=1,
    request_settings=OnlineRequestSettings(
        request_timeout_ms=300000,       # 5 min per request
        max_concurrent_requests_per_instance=1,
        max_queue_wait_ms=300000,
    ),
    liveness_probe=ProbeSettings(
        initial_delay=600,    # 10 min before first health check
        timeout=30,
        period=100,
        failure_threshold=30,
    ),
    readiness_probe=ProbeSettings(
        initial_delay=600,
        timeout=30,
        period=30,
        failure_threshold=30,
    ),
)
deployment = ml_client.online_deployments.begin_create_or_update(deployment).result()

# 3. Route traffic
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
```

### 20.2 Why Probe Settings Matter

Large models take 5–10 minutes to load into GPU memory during `init()`. Azure ML uses liveness and readiness probes to check container health. If a probe fires before `init()` finishes loading the model, Azure concludes the container is unhealthy, kills it, and starts a new one. The new container also takes 5–10 minutes to load, gets killed, and the cycle repeats. The deployment stays in "Creating" state forever with no logs visible.

> *Set `initial_delay` to at least 600 seconds (10 minutes) for any deployment that loads a large model. For 70B+ models, use 900 seconds. This single setting prevents the most common invisible failure mode in GPU deployments. It is not documented prominently and there is no error message that tells you the probe is the problem — the deployment simply never transitions out of "Creating."*

### 20.3 The SDK Object Gotcha

The SDK requires `OnlineRequestSettings` and `ProbeSettings` to be proper SDK objects, not Python dicts. Passing a dict compiles without error and the deployment appears to succeed, but timeout behavior does not work as configured:

```python
# WRONG — dict compiles but fails silently
request_settings={"request_timeout_ms": 180000}

# CORRECT — use the SDK class
request_settings=OnlineRequestSettings(request_timeout_ms=180000)
```

### 20.4 Model Registration

Before deploying, register your model in the workspace:

```python
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

model = Model(
    name="my-model",
    version="1",
    path="/path/to/model/files",
    type=AssetTypes.CUSTOM_MODEL,
    description="Model description",
)
registered = ml_client.models.create_or_update(model)
```

For large models (40GB+), upload to Azure Blob Storage first, then register using the blob path. Direct upload from a compute instance is unreliable for files this size.

---

## 21. Blue-Green Deployments and Traffic Routing

Azure ML supports multiple deployments under a single endpoint. Each deployment runs independently with its own model version, environment, and SKU. Traffic is split by percentage. This enables canary rollouts, A/B testing, and zero-downtime model updates.

```python
# Deploy green alongside existing blue
green = ManagedOnlineDeployment(
    name="green",
    endpoint_name="my-llm-endpoint",
    # ... new model version, new environment ...
)
ml_client.online_deployments.begin_create_or_update(green).result()

# Canary: 10% to green
endpoint.traffic = {"blue": 90, "green": 10}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Full cutover after validation
endpoint.traffic = {"blue": 0, "green": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Remove old deployment to stop billing
ml_client.online_deployments.begin_delete(
    name="blue", endpoint_name="my-llm-endpoint"
).result()
```

You can bypass the traffic split entirely by adding the header `azureml-model-deployment: green` to a request, routing it to a specific deployment for testing before shifting any production traffic.

> *A newly created deployment receives 0% traffic by default. If you deploy and forget to set traffic, the endpoint returns errors for all requests because no deployment is receiving them. Always explicitly set traffic after creating a deployment.*

---

## 22. Multi-Endpoint Orchestrator Architecture

Complex ML pipelines that chain multiple models — summarization feeding into NER feeding into fact-checking — should not be deployed as a single monolithic endpoint. Deploy each model as its own endpoint and use a lightweight CPU orchestrator as the single client-facing entry point.

```
Client → Orchestrator (CPU endpoint)
              ├── LLM Summarizer   (GPU, A100)
              ├── NER Model A       (GPU, T4)
              ├── NER Model B       (GPU, T4)
              └── Entity Linker     (GPU or CPU)
```

The orchestrator receives the client request, fans out HTTP calls to sub-endpoints in parallel using `ThreadPoolExecutor`, collects and merges results, and returns a unified response. It runs on a CPU SKU (`Standard_F8s_v2`, ~$0.34/hr) because it does no inference — only coordination, lightweight embedding, and result formatting.

**Pass endpoint URLs and API keys in the request payload, not as environment variables.** This makes the system reconfigurable without redeploying the orchestrator. Swap a model endpoint, rotate an API key, or add a new NER model — all without touching the running orchestrator container.

Dictionary-based NER and entity linking can run on CPU if latency tolerance is 2–3 seconds per call instead of sub-second. For single-document inference, CPU is fine. For batch processing hundreds of documents, GPU is worth it.

### 22.1 Cost Example

| Component | SKU | Count | Cost/hr |
| --- | --- | --- | --- |
| Orchestrator | F8s_v2 (CPU) | 1 | ~$0.34 |
| LLM (20B params) | NC24ads A100 | 1 | ~$3.67 |
| NER Models | NC4as T4 | 3 | ~$1.59 |
| Entity Linkers | NC4as T4 | 3 | ~$1.59 |
| **Total** | | | **~$7.19** |

Adding a 70B model (AWQ 4-bit on A100) pushes the total to ~$11/hr. Evaluate whether the quality improvement justifies 3–4× the cost of the next tier down — often a distilled 20B model closes enough of the gap.

---

## 23. Debugging Failed Deployments

### 23.1 The "Creating" State Loop

The most frustrating failure mode: the deployment sits in "Creating" for 30+ minutes with no logs. The container has not started, so there is nothing to log. Diagnosis, in order of likelihood:

**Environment build timeout.** Check the build log in your workspace storage account. If the Docker build failed, no container was ever created.

**Probe restart loop.** If `initial_delay` is too short, the container loads for 5 minutes, gets killed by the liveness probe, restarts, loads again, gets killed again. Increase `initial_delay` to 600+ seconds.

**GPU quota exhausted.** The deployment is waiting for capacity that does not exist. Check quota with:

```python
from azure.mgmt.compute import ComputeManagementClient

compute_client = ComputeManagementClient(
    DefaultAzureCredential(), "your-subscription-id"
)
for u in compute_client.usage.list("your-region"):
    if any(k in u.name.localized_value.lower() for k in ["nc24", "a100", "gpu"]):
        print(f"{u.name.localized_value}: {u.current_value}/{u.limit}")
```

**`init()` crashing.** Import errors, CUDA OOM, model file not found. You cannot see these until the container runs long enough to produce logs. Once it does:

```python
logs = ml_client.online_deployments.get_logs(
    name="blue", endpoint_name="my-endpoint", lines=200
)
print(logs)
```

### 23.2 Common Error Patterns

| Error | Root Cause | Fix |
| --- | --- | --- |
| timeout waiting for Environment Image | Conda env too complex for Docker build | Simplify YAML, use pre-built base image |
| ImportError: cannot import name X | Library version mismatch | Verify import exists in pinned version |
| Stuck in Creating indefinitely | init() crash or probe restart loop | Increase initial_delay, add logging to init() |
| run() returns empty response | Returning dict instead of string | Always json.dumps() |
| pydantic version conflict | azureml-inference-server-http pins pydantic | Match pydantic to server requirements |
| CUDA out of memory | Model too large for GPU | Larger SKU or quantize the model |

### 23.3 The Runtime Installation Escape Hatch

When environment builds fail repeatedly — usually complex combinations of CUDA, PyTorch, and vLLM — there is a last-resort technique: use a minimal pre-built environment and install packages inside `init()` at runtime:

```python
import subprocess, sys, os

def init():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "transformers", "vllm", "--quiet"
    ])
    from vllm import LLM
    global llm
    llm = LLM(model=os.environ["AZUREML_MODEL_DIR"])
```

The first request takes 2–3 minutes while packages install. Subsequent requests use the cached installation. This bypasses the Docker build entirely. It is inelegant and should not be standard practice, but when nothing else works, it ships.

---

## 24. Deployment Lessons

These are patterns that emerged from months of production deployment work. None are in the official documentation.

### 24.1 Delete and Recreate; Never Update Failed Deployments

When a deployment fails, calling `begin_create_or_update` on the same deployment usually fails again with stale state — cached Docker layers, partially downloaded models, corrupted container state. The reliable pattern is `begin_delete()`, wait for completion, create fresh. This adds 5 minutes but prevents hours of chasing phantom errors from stale state. Treat failed deployments as disposable.

### 24.2 Register Environments Before Deploying

Do not combine environment registration and deployment into one step. Register the environment first with `ml_client.environments.create_or_update()`, wait for the Docker image to build, confirm success, then deploy. This separates two independent failure modes — environment build and container startup — so you diagnose each one clearly. Combined, a failure could be either, and the error messages do not distinguish.

### 24.3 The code_path Uploads Everything

When you specify `code="./scoring_scripts"` in CodeConfiguration, Azure uploads **every file** in that directory to the deployment container. If the directory contains model weights, datasets, notebooks, or anything large, the upload will be slow or fail. Keep the code directory to scoring scripts and small config files only. Model weights belong in registered Azure ML models, not in the code path.

### 24.4 Test Scoring Scripts Locally Before Deploying

Your Azure ML compute instance has the same GPU that your endpoint will use. Before deploying, copy the scoring script to the compute, import it, call `init()` to load the model, call `run()` with sample JSON. This catches 90% of issues — import errors, model path problems, VRAM limits, serialization bugs — for free, in seconds, with full stack traces. A failed deployment takes 10–30 minutes to surface the same error, if it surfaces at all.

### 24.5 Use Compute Instance Serving for Development

Managed endpoints are for production — when other systems need a stable URL. During development and benchmarking, run the model directly on your compute instance. A vLLM server on localhost or an Ollama instance gives you the same inference quality with zero deployment overhead, zero extra cost beyond the compute itself, and instant iteration. Deploy as a managed endpoint only when you need a stable URL for integration.

### 24.6 Pre-Deployment Checklist

| Check | Verification | Failure If Skipped |
| --- | --- | --- |
| Environment builds | Register separately, wait for Docker build | 30 min in Creating, then failure |
| Imports are valid | `pip show <pkg>` on compute instance | Container crash, no visible logs |
| `run()` returns `json.dumps()` | Test locally with sample input | Empty or malformed responses |
| Model fits GPU | Param count × 2 bytes ≤ VRAM × 0.85 | CUDA OOM, infinite restart loop |
| Probe initial_delay ≥ 600s | Check deployment config | Container killed before load, stuck in Creating |
| `azureml-inference-server-http` in env | Check conda YAML | No HTTP server, all requests fail |
| Traffic set after deployment | Explicitly set `endpoint.traffic` | 0% traffic, endpoint returns errors |

> *The most expensive lesson across all of this: the gap between "model works on a compute instance" and "model works as a managed endpoint" is almost entirely operational, not algorithmic. The model is the same. The math is the same. What changes is the packaging, the probes, the environment builds, the traffic routing, and the debugging visibility. Mastering this operational layer is what separates an ML engineer who builds prototypes from one who runs production systems.*

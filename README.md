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

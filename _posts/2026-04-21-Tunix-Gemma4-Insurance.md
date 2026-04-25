---
layout: post
comments: true
title: Taming the Giant:Fine-Tuning Gemma 4 E2B-IT into an Insurance Expert
published: true
---

In the world of AI, generalists are common, but specialists are rare. Today, I'm sharing the "black box" experience of fine-tuning Google's latest **Gemma 4 E2B-IT** (5.1B parameters) to become a high-precision insurance advisor. 

This wasn't a standard "plug-and-play" exercise. We pushed the boundaries of **Tunix (JAX)** on a **Google Cloud TPU v5litepod-4**, encountered memory walls, patched library-level attention bugs, and learned some hard lessons about Per-Layer Embedding (PLE) tables. Here is the full, unvarnished technical journey.

This project is part of the #TPUSprint. Google Cloud credits are provided for this project.

---

## 1. The Architecture: Why Gemma 4 E2B-IT?

Gemma 4 is a fascinating specimen. While the "E2B" (Edge-to-Base) version is marketed as efficient, its internal architecture is surprisingly complex:
- **Total Parameters:** 5.1B (effectively 2.3B for compute, but with massive embeddings).
- **The PLE Table Factor:** This model uses a **Per-Layer Embedding** table totaling roughly 4.7GB. This allows the model to have a massive vocabulary and nuanced understanding but creates a significant memory footprint that traditional 2B/5B model workflows aren't always prepared for.
- **Context Window:** A massive 128k, though for our InsuranceQA task, we optimized for 1024 to maximize TPU throughput.
- **Chat Template:** It uses specific turn-based tokens: `<|turn|>` (ID: 105) and `<turn|>` (ID: 106), which are critical for maintaining conversational state.

---

## 2. The Infrastructure: TPU v5litepod-4

To handle 5.1B parameters with high-speed JAX sharding, we provisioned a **TPU v5litepod-4**. 
- **Topology:** 2x2 mesh.
- **Memory:** 16GB of High Bandwidth Memory (HBM) per core.
- **Why JAX?** Tunix (Tune-in-JAX) leverages the native XLA compiler to optimize the training graph. On a TPU Pod, JAX sharding (FSDP + Tensor Parallelism) allows us to treat the 4 cores as a single unified compute engine.

---

## 3. Data Engineering: The InsuranceQA-v2 Pipeline

We leveraged the `deccan-ai/insuranceQA-v2` dataset. To make it "Gemma-ready," we built a custom generator that formats every example into a strict turn-based dialogue:

```python
def format_example(example, tokenizer):
    # The Gemma 4 "Secret Sauce" Template
    bos, eos = "<bos>", "<eos>"
    sot, eot = "<|turn|>", "<turn|>"
    
    prompt = f"{bos}{sot}user\n{example['instruction']}{eot}\n{sot}model\n"
    response = f"{example['response']}{eot}{eos}"
    
    # ... tokenization and padding logic ...
```

This formatting ensures the model learns not just the *content* of insurance, but the *style* of a professional advisor responding within a structured turn.

---

## 4. The Training Phase: Convergence and LoRA

We applied **LoRA (Low-Rank Adaptation)** with a Rank of 16 and Alpha of 32. This adds roughly 50MB of trainable parameters to the attention layers (q, k, v, o) while keeping the 5B base weights frozen.

![Training Loss](https://github.com/crownpku/crownpku.github.io/blob/master/images/202604/training_loss.png)

**Key Stats:**
- **Initial Loss:** ~6.99 (The model was essentially guessing).
- **Final Loss:** ~1.2 (Stable convergence after 500 steps).
- **Learning Rate:** 2e-5 with AdamW.
- **Observation:** The loss plummeted within the first 100 steps as the model synchronized with the InsuranceQA-v2 response style.

---

## 5. The "Trough of Disillusionment": Deep-Dive Debugging

This is where the project got real. We hit three major technical walls:

### Wall #1: The 4D Attention Mask Bug
When moving from training to inference on non-standard sequence lengths, Tunix's Gemma 4 implementation threw a `ValueError` in the `einsum` operation. The 3D attention mask wasn't broadcasting correctly to the 4D attention scores.
**The Fix:** We had to dive into `tunix/models/gemma4/model.py` and implement a manual reshape to force 4D broadcasting:
```python
# The Patch that saved the project
expanded_mask = jnp.reshape(attn_mask, (attn_mask.shape[0], -1, 1, attn_mask.shape[-1]))
```

### Wall #2: HBM Exhaustion during LoRA Merging
Gemma 4's 4.7GB PLE table is a beast. When trying to merge LoRA weights back into the base model on the TPU cores, we hit the **16GB HBM limit** immediately.
**The Strategy:** We realized that while TPU cores are for *compute*, the TPU VM's **CPU has 188GB of RAM**. By setting `JAX_PLATFORMS=cpu`, we performed the multi-GB weight merging in host memory, successfully exporting a 9.6GB standalone `.safetensors` file.

### Wall #3: JIT Tracing Errors on CPU
During manual evaluation, JAX's JIT compiler struggled with the module state mutations in the `tunix.generate.Sampler`.
**The Workaround:** We bypassed the JIT-based sampler and wrote a custom token-by-token loop that updated the model state sequentially. This allowed us to verify the model's output quality even without a fully compiled generation graph on CPU.

---

## 6. Quantitative Evaluation: Perplexity Analysis

We didn't rely on "vibe checks" alone. We ran a full perplexity analysis on the test set.

![Perplexity Comparison](https://github.com/crownpku/crownpku.github.io/blob/master/images/202604/perplexity_comparison.png)

Initially, we saw high perplexity (15.8) which we traced back to logit softcapping and broadcasting nuances in the JAX implementation. However, the *relative* improvement was undeniable. The fine-tuned model achieved a 4.2 perplexity on domain-specific questions, signifying a massive increase in confidence and accuracy for insurance queries.

---

## 7. The Result: A Domain Specialist

The qualitative difference was night and day:
- **Scenario:** A user asks about "Split Limits" in auto insurance.
- **Base Model:** Gives a generic definition of liability.
- **Fine-Tuned Model:** Explains the specific 100/300/50 breakdown common in US insurance policies, matching the tone and technical depth of the training data.

---

## 8. Final Takeaways for the AI Developer

1. **JAX is a scalpel:** It is incredibly precise and fast, but you must be ready to manage your own memory sharding and mask broadcasting.
2. **Architecture Matters:** Don't ignore the embedding table size. For models like Gemma 4, the embeddings are half the battle.
3. **CPU is your friend:** For large-model weight merging and export, don't waste HBM; use the high-RAM host CPU.
4. **Validation is Mandatory:** Perplexity metrics and weight statistics (checking means/stds) are the only way to be sure your merged model hasn't degraded.

This journey from a generic E2B model to a specialized Insurance Expert was paved with XLA errors and memory overflows, but the result is a testament to the power of the JAX ecosystem.

**Happy Hacking!** 🚀

---
*Explore the full implementation at: [github.com/crownpku/tunix-gemma4-tpu](https://github.com/crownpku/tunix-gemma4-tpu)*
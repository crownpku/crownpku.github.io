---
layout: post
comments: true
title: "Teaching FunctionGemma to Route Agent Tools Without Writing Essays: Building FunctionGemma-Jev"
published: true
---

# Teaching FunctionGemma to Route Agent Tools Without Writing Essays: Building FunctionGemma-Jev

**Author:** Guan Wang & DeepMind Pair Programming Partner  
**Hardware:** Apple Silicon Unified Memory (MPS)  
**Base Model:** Google FunctionGemma 270M (`unsloth/functiongemma-270m-it` / `google/functiongemma-270m-it`)  
**Repository:** [crownpku/FunctionGemma-Jev](https://github.com/crownpku/FunctionGemma-Jev)

---

## 1. Introduction: The Agent Tool-Calling Bottleneck

Autonomous AI agents—whether handling cloud infrastructure triage, automated CRM updates, or security forensics—live and die by their ability to interact with tools. To support this on edge and developer hardware, Google introduced **FunctionGemma**, a compact, lightweight 270M-parameter model engineered specifically for function calling and tool orchestration.

However, standard FunctionGemma operates like every conventional causal language model: **autoregressively**.

```
[User Query + 10 Verbose Tool Schemas Packed into Prompt]
                         │
                         ▼
             Causal Self-Attention Loop
                         │
                         ├─► Token 1: "<start_function_call>"
                         ├─► Token 2: "call"
                         ├─► Token 3: ":"
                         ├─► Token 4: "restart"
                         ├─► Token 5: "_service" ...
                         │
        (250 – 450 ms of sequential token generation)
```

In a complex multi-turn autonomous agent loop requiring 10 sequential tool steps, this decoding mechanism forces the agent harness to spend **3 to 5 seconds purely waiting for token-by-token character generation**—not for executing business logic, but merely to figure out *which function to trigger*.

### The Three Structural Pathologies of Causal Tool Routing

When we put the original FunctionGemma 270M through systematic empirical stress tests on Apple Silicon, we discovered three structural vulnerabilities:

1. **Severe Positional Order Bias (The 40% Reversal Anomaly):**  
   In autonomous systems (such as Model Context Protocol / MCP registries), tool definitions are registered dynamically; their declaration order in the prompt is completely arbitrary. Yet, when we simply reversed the order of tool declarations in the prompt without changing a single character of the user instruction, the raw autoregressive model **flipped its tool choice in 40.0% of cases**. Causal attention creates an inherent recency or primacy bias that undermines deterministic agent behavior.
2. **Context-Window Tax & KV-Cache Bloat:**  
   To enable causal selection, every candidate function's schema must be concatenated into the input prompt. As registries scale from 5 to 50 tools, prompt token counts explode, inflating memory consumption and KV-cache pressure on edge devices.
3. **Syntax Fragility & Regex Scraping:**  
   Autoregressive generation outputs raw text. If the model hallucinates an unescaped quote, omits a closing brace, or drops a colon, fragile regex parsers (`re.search(r"call:([a-zA-Z0-9_]+)")`) fail, triggering fatal agent crashes mid-task.

Categorical decision-making (selecting 1 tool out of $K$ registered candidates) is fundamentally a **classification and ranking problem**, not an open-ended creative writing exercise. Forcing a causal decoder to compose essays just to output a function name is an architectural mismatch.

---

## 2. Architectural Surgery: Inspired by Jev

Inspired by the non-autoregressive decision principles of **Jev**, we performed surgical architectural modifications directly on the **FunctionGemma 270M transformer trunk**.

Instead of decoding tokens sequentially, we attach two specialized, high-throughput decision heads to the frozen 640-dimensional contextual representations produced by the transformer trunk:
1. **`ToolRouterHead`:** A parallel, candidate-independent bilinear scoring head that evaluates all $K$ tools concurrently.
2. **`ToolSafetyHead`:** A calibrated pre-execution guardrail ("Noul") that evaluates whether executing the chosen tool is autonomous-safe or requires a Human-in-the-Loop (HITL) confirmation prompt.

Here is the high-level architecture comparison:

![](/images/functiongemma_jev_architecture.png)

---

## 3. Dissecting the Decision Surgery Heads

### 3.1. `ToolRouterHead`: Multi-Factor Cross-Projection Scoring

Given a user query $x_{\text{query}}$ and a registry of $K$ candidate tool schemas $\{t_1, t_2, \dots, t_K\}$, we extract mean-pooled representations from the FunctionGemma 270M transformer trunk:

$$h_{\text{state}} = \text{Trunk}(x_{\text{query}}) \in \mathbb{R}^{640}$$
$$h_{\text{tools}} = \text{Trunk}(\{t_1, \dots, t_K\}) \in \mathbb{R}^{K \times 640}$$

Notice that $h_{\text{tools}}$ can be **pre-computed and cached** at tool registration time, reducing runtime compute to just encoding the user query!

Inside `ToolRouterHead`, representations are normalized with dedicated LayerNorm layers and projected into a 256-dimensional decision space. Crucially, we model both independent state/tool features and an element-wise cross-interaction term:

$$z_i = W_{\text{state}} h_{\text{state}} + W_{\text{tool}} h_{\text{tool}_i} + W_{\text{cross}} (h_{\text{state}} \odot h_{\text{tool}_i})$$

Where $\odot$ represents the Hadamard (element-wise) product, capturing semantic alignment between agent intent and tool capability. The interaction vector $z_i \in \mathbb{R}^{256}$ is passed through a scoring MLP:

$$s_i = \text{Linear}_{256 \to 1}(\text{Dropout}(\text{GELU}(z_i))) \in \mathbb{R}$$
$$P(\text{tool}_i) = \text{Softmax}([s_1, s_2, \dots, s_K])_i$$

The winning tool is obtained in constant time via:

$$\text{winner} = \arg\max_i P(\text{tool}_i)$$
$$\text{margin\_confidence} = P(\text{tool}_{(1)}) - P(\text{tool}_{(2)})$$

```python
class ToolRouterHead(nn.Module):
    """
    Candidate-independent parallel tool scoring head.
    Scores each tool definition s_i = Score(h_state, h_tool_i) independently.
    Guarantees 100% permutation invariance across tool registry ordering.
    """
    def __init__(self, hidden_size: int = 640, proj_dim: int = 256):
        super().__init__()
        self.state_norm = nn.LayerNorm(hidden_size)
        self.tool_norm = nn.LayerNorm(hidden_size)
        
        self.state_proj = nn.Linear(hidden_size, proj_dim)
        self.tool_proj = nn.Linear(hidden_size, proj_dim)
        self.cross_proj = nn.Linear(hidden_size, proj_dim)
        
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(proj_dim, 1)
        )

    def forward(self, h_state: torch.Tensor, h_tools: torch.Tensor):
        B, K, D = h_tools.shape
        h_state_norm = self.state_norm(h_state)
        h_tools_norm = self.tool_norm(h_tools)
        
        # Broadcast state across all K candidate tools
        h_state_expanded = h_state_norm.unsqueeze(1).expand(-1, K, -1) # (B, K, D)
        
        # Multi-factor interaction
        z_state = self.state_proj(h_state_expanded)
        z_tool = self.tool_proj(h_tools_norm)
        z_cross = self.cross_proj(h_state_expanded * h_tools_norm)
        
        interaction = z_state + z_tool + z_cross
        logits = self.mlp(interaction).squeeze(-1) # (B, K)
        
        probs = F.softmax(logits, dim=-1)
        winning_index = torch.argmax(probs, dim=-1)
        
        sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)
        margin = sorted_probs[:, 0] - sorted_probs[:, 1] if K > 1 else sorted_probs[:, 0]
        
        return {
            "logits": logits,
            "probs": probs,
            "winning_index": winning_index,
            "margin_confidence": margin
        }
```

#### Why Permutation Invariance is Guaranteed
Because $s_i$ evaluates each pair $(h_{\text{state}}, h_{\text{tool}_i})$ independently without cross-candidate attention leakage, any permutation $\pi$ of the candidate tool registry satisfies:

$$s_{\pi(i)} = \text{Score}(h_{\text{state}}, h_{\text{tool}_{\pi(i)}}) = s_i$$

The scoring is **mathematically equivariant to registry order**, completely eliminating the 40% order-bias flip rate seen in causal models.

---

### 3.2. `ToolSafetyHead`: Calibrated Pre-Execution Guardrail ("Noul")

Agent autonomy requires guardrails. Destructive tools (e.g. `restart_service`, `drop_database`, `rollback_deployment`) should not execute without human confirmation when uncertainty is high or when the context involves critical production systems.

Rather than relying on uncalibrated text prompts (e.g. "are you sure? yes/no"), `ToolSafetyHead` ingests the joint representation of the query and the selected winning tool:

$$h_{\text{joint}} = [h_{\text{state}} \,;\, h_{\text{selected\_tool}}] \in \mathbb{R}^{1280}$$
$$p_{\text{safe}} = \text{Sigmoid}(\text{MLP}_{1280 \to 256 \to 64 \to 1}(\text{LayerNorm}(h_{\text{joint}})))$$

The certainty margin is computed as:

$$\text{certainty} = |2 \cdot p_{\text{safe}} - 1| \in [0.0, 1.0]$$

If $p_{\text{safe}} < 0.70$ (or certainty falls below an operational threshold), the engine halts autonomous execution and surfaces a structured confirmation request to the human operator.

```python
class ToolSafetyHead(nn.Module):
    """
    Binary execution safety guardrail (Noul).
    Evaluates whether calling the selected tool with the requested action is safe
    vs requires human confirmation.
    """
    def __init__(self, hidden_size: int = 640, intermediate_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, h_state: torch.Tensor, h_tool: torch.Tensor):
        combined = torch.cat([h_state, h_tool], dim=-1)
        normed = self.norm(combined)
        prob = self.mlp(normed).squeeze(-1)
        certainty = torch.abs(2.0 * prob - 1.0)
        return {
            "safe_prob": prob,
            "certainty": certainty,
            "requires_confirmation": (prob < 0.70)
        }
```

---

## 4. Empirical Benchmarks on Apple Silicon Unified Memory

We evaluated both models locally on an Apple Silicon machine (M-series MPS backend) across realistic multi-domain autonomous agent scenarios covering DevOps incident mitigation, CRM record modifications, network triage, and conversational no-tool queries.

### Head-to-Head Quantitative Results

| Evaluation Metric | Raw FunctionGemma 270M (Autoregressive) | FunctionGemma-Jev (Ours) | Advantage / Gain |
| :--- | :--- | :--- | :--- |
| **Tool Selection Accuracy** | **65.0%** | **100.0%** | **+35.0% Accuracy Parity Gain** |
| **Tool Routing Head Latency** | **247.5 ms** (Token decoding loop) | **2.84 ms** (Parallel tensor pass) | **87x Faster Tool Routing** |
| **Total Pipeline Latency** | **247.5 ms** | **49.87 ms** (Trunk + Head) | **5.0x Faster End-to-End** |
| **Order Bias (Prompt Flip Rate)** | **40.0%** (Order sensitive) | **0.0%** (Mathematically invariant) | **Complete Permutation Invariance** |
| **Syntax & Parsing Crashes** | Vulnerable to malformed brackets | **0.00%** (Direct tensor float) | **Zero Schema Crashes** |
| **Pre-Execution Guardrail** | Uncalibrated string generation | **Calibrated Probability + Margin** | **Native Human-in-the-Loop** |

### Latency Deep Dive

```
Raw FunctionGemma 270M (Autoregressive Causal Decoding):
████████████████████████████████████████████████ 247.5 ms

FunctionGemma-Jev (Total Pipeline: Trunk Embedding + Decision Surgery):
█████████▍ 49.87 ms  (5.0x faster end-to-end)

FunctionGemma-Jev (ToolRouterHead Readout Only):
▍ 2.84 ms  (87x faster decision readout)
```

When candidate tool embeddings $h_{\text{tools}}$ are pre-computed at application startup (which is standard practice in fixed agent registries), the runtime routing overhead drops to **under 3 milliseconds**.

---

## 5. Running the Live Comparison

We built an interactive Gradio web application that runs both engines side by side on Apple Silicon MPS:

```bash
# 1. Install dependencies via uv
uv sync

# 2. Run the head-to-head empirical benchmark
.venv/bin/python eval_tool_calling.py

# 3. Launch the interactive Gradio comparator
.venv/bin/python app.py
```

The UI displays real-time execution telemetry, candidate probability distributions, margin confidence scores, and safety guardrail verdicts with instant permutation stress-testing.

---

## 6. Key Takeaways & What This Means for Agent Systems

1. **Decouple Tool Routing from Argument Decoding:**  
   Language models should not waste compute decoding tokens sequentially just to make a discrete routing decision. By isolating tool selection into a non-autoregressive decision head, agents achieve **sub-3ms routing speeds**.
2. **Order Invariance is Non-Negotiable for Autonomous Registries:**  
   In agent architectures using dynamic protocols like MCP, tool order changes across sessions. Relying on causal prompts introduces a 40% non-deterministic flip risk. Non-autoregressive bilinear scoring guarantees mathematical order invariance.
3. **Calibrated Safety Out-of-the-Box:**  
   Attaching a dedicated binary classification head directly to the transformer representations gives developers calibrated confidence intervals ($p_{\text{safe}}$ and certainty margins), providing a principled trigger for human-in-the-loop approvals.
4. **Edge Feasibility:**  
   A 270M parameter trunk running on Apple Silicon unified memory can route tools with 100% accuracy and 50ms total latency, making autonomous agents fully viable on edge hardware and local laptops.

---

*The full implementation, checkpoints, and benchmark datasets are open-sourced in the [`FunctionGemma-Jev`](https://github.com/crownpku/FunctionGemma-Jev) repository.*

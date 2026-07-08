# AI Guide — Auto Research Agent (AutoResearcher)

> **This file is designed to be read by AI assistants (Claude, ChatGPT, Codex, etc.).**
> When a user pastes this file or asks you to help them set up Auto Research Agent,
> follow the instructions below to guide them step by step.
>
> This guide reflects the current state of the repo (`config.yaml`, `core/`, `api.py`, `docs/architecture_CN.md`).
> Default provider: Zhipu GLM (`glm_token_plan`) with auto-failover to Ali.

---

## What Is This?

Auto Research Agent (AutoResearcher) is a framework that lets an AI agent autonomously run deep learning experiments 24/7. Given a research brief and a dataset, it end-to-end handles: understanding the data, surveying methods, designing experiments, implementing & training models, **verifying** results, and reflecting to iterate.

The core loop (`core/loop.py`, `ResearchLoop.run()`):

```
THINK → EXECUTE → VERIFY → REFLECT → repeat
```

1. **Fact Spine** — scans disk for real experiment facts *before any LLM call* (ground truth)
2. **THINK** — Leader LLM analyzes state, forms a falsifiable hypothesis, plans the experiment
3. **EXECUTE** — code agent writes/modifies code, dry-runs, launches GPU training
4. **MONITOR** — watches training at ZERO LLM cost (only process checks + log reads)
5. **VERIFY** — 12-layer reverse-engineering check + anti-cheat (results must be backed by tool traces)
6. **REFLECT** — evaluate, update memory, record `dead_end`s / lessons, decide next step

**Killer feature:** during training (90%+ of wall-clock), the agent makes ZERO API calls — it only does `kill -0 $PID`, `nvidia-smi`, and `tail` on the log. Cost is driven by the THINK/REFLECT calls only.

---

## Your Job as AI Assistant

When a user asks for help with this project, follow this decision tree:

```
User wants to...
├── Install it → [SETUP GUIDE]
├── Create a project → [PROJECT CREATION]
├── Launch the agent → [LAUNCH GUIDE]
├── Check status → [STATUS CHECK]
├── Intervene/redirect → [INTERVENTION]
├── Use on phone → [MOBILE SETUP]
├── Understand how it works → [ARCHITECTURE EXPLANATION]
└── Debug an issue → [TROUBLESHOOTING]
```

---

## SETUP GUIDE

### Step 1: Check Prerequisites

```bash
python3 --version                 # Need 3.10+
nvidia-smi                        # Need at least 1 GPU
echo $GLM_CODING_PLAN_API_KEY     # Default provider: Zhipu GLM
```

If Python < 3.10: suggest `conda create -n dra python=3.11 -y && conda activate dra`

If no GPU: this framework requires a GPU for training. Suggest cloud GPU (Lambda Labs, RunPod, Vast.ai).

If no GLM key: obtain one from Zhipu AI (智谱) Coding Plan, then:
```bash
export GLM_CODING_PLAN_API_KEY="your-key-here"
echo 'export GLM_CODING_PLAN_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

> Other providers are also supported (see Step 3). If the user prefers Anthropic/OpenAI/Ali, set the corresponding key instead.

### Step 2: Install

```bash
pip install -r requirements.txt   # anthropic / openai / zai-sdk / pyyaml
python install.py                 # Install Claude Code / Codex skills
python api.py --help              # Verify the CLI is wired up
```

`python install.py` installs the slash-command skills (e.g. `/auto-experiment`, `/experiment-status`, `/gpu-monitor`, `/daily-papers`, `/paper-analyze`, `/conf-search`, `/progress-report`, `/obsidian-sync`, plus research/audit skills). Run `python install.py --uninstall` to remove them.

### Step 3: Choose Your LLM Provider

Default is **`glm_token_plan` (Zhipu GLM)**, with auto-failover to `ali_token_plan` on quota errors. To switch, edit `config.yaml`:

| Provider | Strong Model | Fast Model | Env Var |
|----------|-------------|-----------|---------|
| `glm_token_plan` *(default)* | glm-5.1 (thinking auto) | glm-5 | `GLM_CODING_PLAN_API_KEY` |
| `ali_token_plan` | qwen3.6-plus | deepseek-v3.2 | `ALI_TOKEN_PLAN_API_KEY` |
| `qwen` | qwen3.6-plus | qwen3.6-plus | `ALI_API_KEY` |
| `anthropic` | claude-opus-4-6 | claude-sonnet-4-6 | `ANTHROPIC_API_KEY` |
| `openai` | gpt-5.4 | codex-5.3 | `OPENAI_API_KEY` |

```yaml
agent:
  provider: "glm_token_plan"   # or anthropic / openai / qwen / ali_token_plan
  model: "auto"                # auto = strong model for think/reflect, fast for code/writing
```

> Note (from `config.yaml`): as of 2026-06, `glm-5.2` returns 403 on standard Coding Plan tiers, so the strong model is `glm-5.1` until authorized. The failover chain handles this automatically.

---

## PROJECT CREATION

### Ask the User These Questions:

1. **What's your research goal?** (specific metric + target, e.g. "ViT on CIFAR-100 to 85% accuracy")
2. **Do you already have training code?** (Yes → point to it / No → agent will create it)
3. **Where is your data?** (path or "auto-download")
4. **Which GPU(s) can you use?** (run `nvidia-smi`)
5. **Any constraints?** (max epochs, batch size, etc.)

### Create the Project Directory:

```bash
mkdir ~/PROJECT_NAME
cd ~/PROJECT_NAME
```

### Write PROJECT_BRIEF.md (Tier 1 memory — frozen, agent never edits it)

```markdown
# Goal
[Specific metric + target value]

# Codebase
[If existing code: list files and paths]
[If no code: "Agent should create PyTorch training code from scratch"]
- Data: [path or "auto-download via torchvision"]
- Checkpoints: ./checkpoints/
- Logs: ./logs/

# What to Try
[Decision tree based on the user's domain knowledge]
- First try: [baseline config]
- if [metric] < [threshold1]: try [approach A]
- if [metric] between [threshold1] and [threshold2]: try [approach B]
- if [metric] > [target]: goal reached, generate report

# Constraints
- GPU: [which GPU(s)]
- Max epochs per run: [number]
- Batch size: [number]
```

### Write a minimal config.yaml (optional — repo default works too)

```yaml
project:
  name: "PROJECT_NAME"
  brief: "PROJECT_BRIEF.md"

goals:                       # the agent uses this to decide "goal reached"
  metrics:
    - key: "val_accuracy"    # or val_MAE, etc.
      target: 0.85
      direction: "higher"    # or "lower"
  stop_on_achieved: true

agent:
  provider: "glm_token_plan"
  model: "auto"
```

### Key Tips to Tell the User:

- **Be specific about the goal** — "accuracy > 80%", not "improve accuracy"
- **Give a decision tree** — the agent needs to know what to do in each situation
- **Keep PROJECT_BRIEF under 3000 characters** — this is the Tier 1 memory cap
- **Think of it as instructing a capable but new PhD student**

See `examples/toy_experiment/PROJECT_BRIEF.md` for a minimal working example (MNIST CNN).

---

## LAUNCH GUIDE

### Option A: Friendly CLI — `api.py` (recommended)

```bash
python api.py start --project ~/PROJECT_NAME --gpu 0 --max-cycles -1   # background daemon (24/7)
python api.py run   --project ~/PROJECT_NAME --cycles 1                # synchronous, single cycle (good for first run)
python api.py status --project ~/PROJECT_NAME                          # check status
python api.py lessons --project ~/PROJECT_NAME --severity HIGH         # view dead_ends / lessons
python api.py stop   --project ~/PROJECT_NAME                          # stop the daemon
```

### Option B: Core loop directly

```bash
nohup python -m core.loop \
  --project ~/PROJECT_NAME \
  --gpu 0 \
  --max-cycles 5 \    # optional: omit / use -1 for unlimited
  > loop.log 2>&1 &
```

### Option C: Claude Code / Codex skill

```
/auto-experiment --project ~/PROJECT_NAME --gpu 0
```

### What to Tell the User:

"The agent is now running. Here's what will happen:
1. It reads PROJECT_BRIEF.md
2. Scans disk for prior experiment facts (Fact Spine)
3. Plans the first experiment with a falsifiable hypothesis
4. Writes/modifies code and does a dry-run to catch errors
5. Launches real training
6. During training: ZERO API cost — it just checks if the process is alive
7. When training finishes, it runs the 12-layer VERIFY + methodology gates, then REFLECTs
8. This repeats until you stop it or the goal is reached

You can close this terminal — training continues via nohup. Check back with `/experiment-status` or `python api.py status`."

---

## STATUS CHECK

```bash
# Friendly CLI:
python api.py status --project ~/PROJECT_NAME
python api.py lessons --project ~/PROJECT_NAME --severity HIGH

# Skills:
/experiment-status --project ~/PROJECT_NAME
/gpu-monitor

# Manual:
cat ~/PROJECT_NAME/workspace/MEMORY_LOG.md      # milestones, recent decisions, dead_ends (human-readable)
cat ~/PROJECT_NAME/workspace/.cycle_counter     # cycles completed
nvidia-smi                                       # GPU usage
```

For persistent progress notes, enable Obsidian sync in `config.yaml`:

```yaml
obsidian:
  enabled: true
  vault_path: "~/Documents/MyObsidianVault"    # optional; empty → local fallback
  project_subdir: "DeepResearcher/{project_name}"
  auto_append_daily: true
```

- If `vault_path` is set, writes `Dashboard.md` + daily Markdown notes into that vault.
- If empty, falls back to project-local files under `workspace/progress_tracking/`.
- Manual refresh: `/obsidian-sync --project ~/PROJECT_NAME` or `python -m core.obsidian --project ~/PROJECT_NAME`.

---

## INTERVENTION

The user wants to change the agent's direction. Three methods:

### Method 1: Directive File (Recommended)
```bash
echo "YOUR INSTRUCTION HERE" > ~/PROJECT_NAME/workspace/HUMAN_DIRECTIVE.md
```
The agent reads this at the start of the next cycle with HIGHEST priority, then auto-archives it.

Examples:
- `"Stop trying ResNet. Switch to ViT-B/16 with lr=1e-3"`
- `"The last 3 experiments all used lr=0.1. Try smaller: 1e-3, 1e-4, 1e-5"`
- `"Goal reached! Generate a final report with all results."`

### Method 2: Command-Line
```bash
python -m core.loop --project ~/PROJECT_NAME --directive "Try label smoothing 0.1"
```

### Method 3: Edit Memory
```bash
vim ~/PROJECT_NAME/workspace/MEMORY_LOG.md
```
For permanent information injection — the agent reads this every cycle.

---

## MOBILE SETUP

For checking experiments from a phone:

```bash
npm install -g happy-coder      # Install Happy Coder CLI
happy                           # Start session through Happy

# Inside: launch experiment
/auto-experiment --project ~/PROJECT_NAME --gpu 0
```

Then install the Happy Coder app:
- iOS: https://apps.apple.com/us/app/happy-codex-claude-code-app/id6748571505
- Android: https://play.google.com/store/apps/details?id=com.ex3ndr.happy

Scan the QR code to pair. The user then gets push notifications and can send directives from the phone.

---

## ARCHITECTURE EXPLANATION

Use this when the user asks "how does it work?" Full detail: `docs/architecture_CN.md`.

### Three-layer design philosophy

| Layer | Role | Implemented in |
|---|---|---|
| **System = hard constraints** | safety, lifecycle, tools, memory, methodology — *the LLM cannot bypass these* | `core/tools.py`, `core/verifier.py`, `core/methodology_gates.py`, `core/constraint_engine.py` |
| **Guidance = research methodology** | *how* to think (hypothesis, controls, falsification), not *what* to do | `agents/*.md` prompts |
| **LLM = the PhD brain** | design, implement, judge, iterate | the model (GLM / Qwen / Claude / GPT) |

### The Loop
```
Fact Spine → THINK → EXECUTE → MONITOR (zero-LLM) → VERIFY → Methodology Gates → REFLECT → GC → next cycle
```

### Why It's Cheap
During training (90%+ of time), the agent does NOT call the LLM. It only does:
- `kill -0 $PID` — is the process alive? (zero cost)
- `nvidia-smi` — is GPU active? (zero cost)
- `tail -50 logfile` — latest metrics (zero cost)

### Memory System (3 tiers)
- **Tier 1**: `PROJECT_BRIEF.md` — frozen, human-written, max 3000 chars
- **Tier 2**: `workspace/MEMORY_LOG.md` — rolling, auto-compacted (milestones + recent decisions + dead_ends), ~4000 chars
- **Tier 3**: `workspace/experiment_history.db` (SQLite) — full history: `experiments`, `memory_entries`, `causal_chain`, `code_review_lessons`, `pareto_matrix`, `experiment_value`, `experiment_facts`

The LLM can actively query history via the **`query_memory`** tool (causal chains, dead_ends, best metric).

### Multi-Agent Architecture (Leader-Worker)
Only **one worker** runs at a time; the others cost $0.
- **Leader** (strong model): THINK + REFLECT decisions
- **code agent** (fast model): implement + train
- **idea / researcher agent** (strong model): literature + deep search (multimodal)
- **writing agent**: reports

### Safety & Integrity (the system's soul — runs at the tool/fact layer, LLM cannot talk its way past)
- **Tool-layer safety** (`core/tools.py`): protected files/dirs, `run_python` blacklist (`os.system`/`subprocess`/`eval`), shell-command validation (~30 patterns, blocks `rm -rf /`, reverse shells), path sandbox, mandatory dry-run gate, `experiment_manifest.json` evidence
- **12-layer VERIFY** (`core/verifier.py`): anti-cheat — results must be backed by tool traces, not LLM claims
- **Methodology gates G1–G4** (`core/methodology_gates.py`): falsifiability, control-coverage, dead-end signature, spec-conformance
- **Constraint engine** (`core/constraint_engine.py`): turns history into executable rules (a method marked dead_end 5+ times becomes `forbidden` and is hard-blocked at `launch_experiment`)
- **Provider failover**: GLM → Ali, quota-aware cooldown
- **Crash-recoverable**: `cycle_count` saved at cycle start; `state.json` written atomically (tmp + rename)

### One source of truth per data type
e.g. `dead_end` records live only in `memory_entries` (`entry_type='dead_end'`), never duplicated across tables — enforced by the L3 DB read/write contract test. See `docs/DATA_CONTRACT.md`.

---

## TROUBLESHOOTING

### "No GPU found"
```bash
nvidia-smi   # Check if CUDA drivers are installed
```
If not: install NVIDIA drivers for the GPU.

### "zai-sdk / anthropic / openai package not found"
```bash
pip install -r requirements.txt   # installs the right SDK for the chosen provider
```

### "API key not set"
```bash
export GLM_CODING_PLAN_API_KEY="your-key-here"    # default provider
# or, for other providers:
export ANTHROPIC_API_KEY="..."   /   export OPENAI_API_KEY="..."   /   export ALI_TOKEN_PLAN_API_KEY="..."
```

### "Dry-run failed"
This is working as intended! The dry-run caught an error before wasting GPU hours. Check the error and fix the code, or let the agent fix it in the next cycle.

### "Agent keeps trying the same thing"
Drop a directive:
```bash
echo "You've tried X three times. Try something completely different: Y" \
  > workspace/HUMAN_DIRECTIVE.md
```
The constraint engine will also auto-escalate repeated dead_ends to `forbidden`.

### "Training crashed"
The agent detects crashes (PID dies), reads the error log, and tries to fix the issue. If it keeps crashing, intervene with a directive.

### "Memory is full / context too long"
Memory is capped. If it happens, check:
```bash
wc -c PROJECT_BRIEF.md                # should be < 3000
wc -c workspace/MEMORY_LOG.md         # should be < 4000
```

### "Want to start fresh"
```bash
rm -rf workspace/                      # delete all agent state (PROJECT_BRIEF.md is preserved)
python api.py run --project . --cycles 1    # restart from scratch
```

---

## QUICK REFERENCE

| Command | What It Does |
|---------|-------------|
| `python api.py start --project PATH --gpu 0 --max-cycles -1` | Start 24/7 background daemon |
| `python api.py run --project PATH --cycles 1` | Run N cycles synchronously |
| `python api.py status --project PATH` | Get project status |
| `python api.py lessons --project PATH --severity HIGH` | Show learned lessons / dead_ends |
| `python api.py stop --project PATH` | Stop the daemon |
| `/auto-experiment --project PATH --gpu 0` | Start loop via skill |
| `/experiment-status` | Check progress |
| `/gpu-monitor` | GPU status |
| `/daily-papers` | Paper recommendations |
| `/paper-analyze ARXIV_ID` | Deep paper analysis |
| `/conf-search --venue CVPR2025 --query "xxx"` | Conference search |
| `/progress-report` | Generate report |
| `echo "instruction" > workspace/HUMAN_DIRECTIVE.md` | Redirect agent |
| `python install.py --uninstall` | Remove all skills |

---

## IMPORTANT CONTEXT FOR AI ASSISTANTS

- This framework works with ANY training framework (PyTorch, TensorFlow, JAX, etc.)
- The agent can create code from scratch OR modify existing code
- It's not just hyperparameter tuning — it can change architectures, loss functions, augmentation strategies
- The agent is NOT a chatbot — it's an autonomous loop that runs independently
- During training it makes ZERO LLM calls, so daily cost is far below an always-on-LLM approach
- If the user is confused, start with: "Let's create a simple project first and watch the agent work" (try `examples/toy_experiment/`)

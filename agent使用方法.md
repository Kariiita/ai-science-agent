# Auto Research Agent 使用方法

> 一份基于**当前代码与文档真实状态**（`docs/README_CN.md`、`docs/architecture_CN.md`、`config.yaml`、`core/loop.py`、`api.py`）的详细使用指南。
>
> 默认 LLM provider 为**智谱 GLM**（`glm_token_plan`），需要 `GLM_CODING_PLAN_API_KEY`；也支持 Anthropic / OpenAI / Qwen / 阿里 Token Plan，改 `config.yaml` 的 `provider` 即可切换。更全面的 AI 协作指南见项目根 [`CLAUDE.md`](CLAUDE.md)。

---

## 一、它是什么

**Auto Research Agent** 是一个能 7×24 小时**全自动**跑深度学习实验的智能体框架。你给它一份研究目标（brief）和数据集，它会自主完成：

> **理解数据 → 调研方法 → 设计实验 → 写代码训练 → 验证结果 → 反思迭代 → 循环往复**

核心循环（`core/loop.py` 中 `ResearchLoop.run()`）：

```
THINK → EXECUTE → VERIFY → REFLECT → 循环
```

**最关键的成本特性**：训练期间（占 90%+ 时间）**完全不调 LLM**，只做"进程还活着吗 + nvidia-smi + 读日志"三件事。所以一天成本约几毛钱，而不是几十美元。

---

## 二、前置准备

| 项 | 要求 |
|---|---|
| Python | 3.10+ |
| GPU | 至少 1 张 NVIDIA GPU（训练用） |
| API Key | 默认需要**智谱 GLM**：`GLM_CODING_PLAN_API_KEY` |

设置 key（以默认的 GLM 为例）：

```bash
export GLM_CODING_PLAN_API_KEY="your-key-here"
# 永久生效：
echo 'export GLM_CODING_PLAN_API_KEY="your-key-here"' >> ~/.bashrc && source ~/.bashrc
```

> 没有智谱 key？也支持 Anthropic (`ANTHROPIC_API_KEY`)、OpenAI (`OPENAI_API_KEY`)、阿里 (`ALI_API_KEY`)、阿里 Token Plan (`ALI_TOKEN_PLAN_API_KEY`)。改 `config.yaml` 中的 `provider` 字段即可。

---

## 三、安装

```bash
pip install -r requirements.txt          # 依赖：anthropic / openai / zai-sdk / pyyaml
python install.py                        # 安装 Claude Code skills（/auto-experiment 等）
python api.py --help   # 验证可运行（应列出 status/run/start/stop/lessons 子命令）
```

---

## 四、创建项目（最重要的一步）

一个项目目录需要三样东西：

### 1. `PROJECT_BRIEF.md` —— 研究目标（冻结，<3000 字符）

这是**整个系统的灵魂文件**，agent 永不修改它。要像"指导一个能力强但刚入门的博士生"那样写：目标具体、给出决策树、写明约束。

参考仓库自带的示例 `examples/toy_experiment/PROJECT_BRIEF.md`：

```markdown
# 目标
在 CIFAR-100 上训练 ViT，验证准确率达到 85%。

# 代码库
- 训练脚本：train.py（agent 自行创建）
- 数据：torchvision 自动下载
- 检查点：./checkpoints/
- 日志：./logs/

# 要尝试什么（决策树）
- 先试：ViT-B/16, lr=1e-3, 50 epoch（baseline）
- 若准确率 < 60%：检查学习率，尝试 5e-4 / 1e-4
- 若 60% < 准确率 < 80%：加数据增强 + mixup
- 若准确率 > 85%：达标，生成报告

# 约束
- GPU：0
- 每次最多 50 epoch
- batch size ≤ 256
```

### 2. 数据集

放进项目目录，或在 brief 里写"auto-download via torchvision"。

### 3. `config.yaml`（可选，不写则用仓库默认）

至少指定目标和 provider：

```yaml
project:
  name: "my_experiment"
  brief: "PROJECT_BRIEF.md"

goals:                    # agent 靠它判断"达标没有"
  metrics:
    - key: "val_accuracy"
      target: 0.85
      direction: "higher"   # 或 "lower"（如 val_MAE）

agent:
  provider: "glm_token_plan"
  model: "auto"             # auto = think/reflect 用强模型，code/writing 用快模型
```

完整配置项见 `config.yaml` 的注释。

---

## 五、启动

先理解一个**关键区别**——同一套研究方法论有两条执行路径：

| | python 方式（方式 A / B） | skill 方式（方式 C） |
|---|---|---|
| 循环驱动者 | 独立 Python 进程（`core/loop.py`） | Claude Code 会话 |
| LLM 大脑 | 调 API（GLM/Claude，用 API key） | Claude Code 自己（用 CC 额度） |
| 硬约束 / 反欺骗 | **代码强制**（12 层 VERIFY、方法论门、constraint_engine），LLM 无法绕过 | 退化为 SKILL.md 文字提示，靠 Claude 自觉 |
| 关掉终端/会话 | **继续跑**（nohup 独立进程） | 循环停止（当前这轮训练会跑完） |
| 适合场景 | 长期 24/7 无人值守、严肃实验 | 边盯边调、快速验证 |

> 一句话：想关终端长期跑 → 用 python 方式；想在 Claude Code 里交互式驱动 → 用 skill 方式。两者都会把训练脚本（`train.py`）真实启动到 GPU 上，区别只在"谁来当大脑、谁来驱动循环"。

### 方式 A：友好 CLI（python 方式，推荐入门）—— `api.py`

`api.py` 是封装好的命令行入口：

```bash
# 跑一个周期（同步，便于先看效果）
python api.py run --project ~/my_experiment --cycles 1

# 后台 daemon 常驻（7×24，关终端继续跑）
python api.py start --project ~/my_experiment --gpu 0 --max-cycles -1

# 查看状态
python api.py status --project ~/my_experiment

# 查看积累的经验教训（dead_end / lessons）
python api.py lessons --project ~/my_experiment --severity HIGH
```

### 方式 B：直接跑核心循环（python 方式）

```bash
nohup python -m core.loop \
  --project ~/my_experiment \
  --gpu 0 \
  --max-cycles 100 \
  > loop.log 2>&1 &
```

独立进程，关闭终端也会继续跑（nohup）。

### 方式 C：在 Claude Code 里用 skill（skill 方式）

**前置**：在项目目录里用 Claude Code 时 skill 已自动识别；想全局可用（任何项目都能用）则跑：

```bash
python install.py --claude-code   # 装到 ~/.claude/commands/
```

在 Claude Code 对话里：

```
/auto-experiment --project ~/my_experiment --gpu 0
/auto-experiment --project . --max-cycles 5
```

参数：`--project <路径>` / `--gpu <id>` / `--max-cycles <n>`（不写 = 无限）。

Claude 会按 `skills/auto-experiment/SKILL.md` 剧本执行 THINK→EXECUTE→REFLECT：自己读 brief、写代码、用 `nohup` 启动训练、`tail` 日志、更新 memory。**训练进程会被真实启动到 GPU**，但**循环由 Claude Code 会话驱动**——所以**关掉会话循环就停**（当前这轮 nohup 训练仍会跑完）。要 24/7 无人值守，请用方式 A/B。

---

## 六、监控与查看状态

```bash
# 实验结果与决策历史（人读）
cat ~/my_experiment/workspace/MEMORY_LOG.md

# 已经跑了多少周期
cat ~/my_experiment/workspace/.cycle_counter

# GPU 占用
nvidia-smi
```

**三层记忆系统**（`docs/architecture_CN.md` 第 5 节）：

| 层 | 文件 | 特点 |
|---|---|---|
| Tier 1 | `PROJECT_BRIEF.md` | 冻结，人工写，<3000 字符 |
| Tier 2 | `workspace/MEMORY_LOG.md` | 滚动，LLM 每周期读，上限 ~4000 字符 |
| Tier 3 | `workspace/experiment_history.db` | SQLite 全历史（experiments / memory_entries / causal_chain 等） |

想看进度可视化，可在 `config.yaml` 中开启 Obsidian 同步（自动生成 Dashboard.md + 每日笔记）。

---

## 七、人工干预 / 重定向方向

agent 跑偏了？三种方式介入：

### 方法 1：指令文件（推荐，下一周期自动读取并归档）
```bash
echo "别再试 ResNet 了。改用 ViT-B/16，lr=1e-3" \
  > ~/my_experiment/workspace/HUMAN_DIRECTIVE.md
```

### 方法 2：命令行参数
```bash
python -m core.loop --project ~/my_experiment --directive "试 label smoothing 0.1"
```

### 方法 3：直接编辑 `MEMORY_LOG.md`
适合永久性信息注入（agent 每周期都读）。

---

## 八、架构速览（理解它"为什么靠谱"）

### 多 Agent（Leader-Worker，同时只跑 1 个 worker，省 token）
- **Leader**（强模型）：决策"做什么"、反思结果
- **code agent**（快模型）：写代码、启动训练
- **idea / researcher agent**（强模型）：查文献、深度搜索
- **writing agent**：生成报告

### 硬约束 = 系统的灵魂（LLM 无法绕过）
这套机制跑在**工具/事实层**而非 LLM 层，是防止"agent 嘴上说达标、实际没达标"的关键（`docs/architecture_CN.md` 第 6 节）：

- **工具安全**：受保护文件/目录、`run_python` 黑名单（禁止 `os.system`/`subprocess`/`eval`）、shell 命令校验（拦 `rm -rf /`、反弹 shell）、路径沙箱、强制 dry-run 门
- **12 层 VERIFY**：反欺骗——结果必须由工具痕迹佐证，LLM 编不出来
- **方法论门 G1–G4**：可证伪性、对照覆盖、死端签名、规格符合
- **dead_end 闭环**：被证伪的方法记录进 `memory_entries`，重试 5 次自动变 `forbidden` 硬阻断
- **崩溃可恢复**：周期计数器每周期开始即存盘，`state.json` 原子写入

> ⚠️ **以上硬约束是 python 方式（`core/loop.py`）专属**——Python 代码强制执行，LLM 无法绕过。**skill 方式**（方式 C）下这些大多**退化为 `SKILL.md` 的文字提示**（靠 Claude 自觉，无代码强制），记忆也只有两层（无 SQLite 全历史、无 `query_memory`）。两者是"同一方法论的两种实现"，**不是等价物**——要严肃实验 / 防自欺用 python 方式，快速验证 skill 方式够用。

### Provider 故障转移
GLM 主用 → Ali 兜底，配额耗尽自动冷却切走（`config.yaml` 注释有说明）。

---

## 九、常见问题

| 现象 | 处理 |
|---|---|
| `No GPU found` | 确认装了 NVIDIA 驱动：`nvidia-smi` |
| API key 未设置 | `export GLM_CODING_PLAN_API_KEY="..."` |
| Dry-run 失败 | **这是设计如此**——错误在浪费 GPU 之前被抓到了，让 agent 下个周期修 |
| agent 反复试同一招 | 丢指令文件："你已经试过 X 三次了，试完全不同的 Y" |
| 想从头来过 | `rm -rf workspace/`（`PROJECT_BRIEF.md` 会保留），重启即可 |

---

## 快速参考表

| 命令 | 作用 |
|---|---|
| `python api.py run --project PATH --cycles 1` | 同步跑 1 个周期 |
| `python api.py start --project PATH --gpu 0 --max-cycles -1` | 后台 daemon 常驻 |
| `python api.py status --project PATH` | 查看状态 |
| `python api.py lessons --project PATH --severity HIGH` | 查经验教训 |
| `/auto-experiment --project PATH --gpu 0` | Claude Code skill 启动 |
| `/experiment-status` | 查进度（skill） |
| `/gpu-monitor` | GPU 状态（skill） |
| `/progress-report` | 生成报告（skill） |
| `echo "指令" > workspace/HUMAN_DIRECTIVE.md` | 重定向 agent |
| `python install.py --uninstall` | 卸载所有 skills |

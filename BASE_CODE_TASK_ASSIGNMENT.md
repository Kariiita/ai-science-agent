================================================================================
V-SciAgent 视研 基座代码（auto-research-agent）修改任务书
================================================================================
发给人：后端/算法同学
编写日期：2026-07-11
项目：阿里云大学生挑战杯 V-SciAgent 视研
数据集：NYU Depth v2（先用这个）
基座代码位置：D:\code\auto-research-agent\
验证场景：D:\code\auto-research-agent\depth_project\

本文档总结基座代码当前存在的全部问题、需要进行的修改、以及具体操作流程。
按优先级分为 P0（阻塞闭环）/ P1（影响拿奖质量）/ P2（打磨提升）三级。
================================================================================


一、现状诊断：什么能用，什么不能用
================================================================================

【能用的部分（不需要改）】
  - 核心闭环引擎 core/loop.py（THINK -> EXECUTE -> VERIFY -> REFLECT 全流程）
  - 多智能体调度 core/agents.py（Leader/Researcher/Idea/Code/Writing 五个 agent）
  - 反欺骗 ToolTrace（core/agents.py 中的 trace 记录机制）
  - 方法论闸门 core/methodology_gates.py
  - 12 层验证 core/verifier.py
  - 死胡同学习 core/constraint_engine.py
  - 记忆管理 core/memory.py + core/fact_scanner.py
  - 实验监控 core/monitor.py
  - Qwen/DashScope API 连接（日志已确认 200 OK，连接正常）
  - DORN baseline 模型 scripts/dorn.py（基本可用，需补评估指标）
  - GPU：RTX 5060 Laptop 8GB，Python 3.12.4

【不能用的部分（必须改）】
  P0 级（阻塞闭环运行）：
    1. 数据集：data/ 目录是空的，只有 README.txt
    2. 数据加载器：datasets/ 目录是空的，train_dorn.py 导入的
       from datasets.nyu_depth_v2 import NYUDepthV2 会直接报 ImportError
    3. 指标体系：config.yaml 写的是 val_MAE target 0.20，
       但深度估计用的是 AbsRel/RMSE/delta1，不是 MAE
    4. 训练脚本只算 L1 loss，不算 AbsRel/RMSE/delta1，
       agent 无法判断是否在改善
    5. 文献搜索全部失败：web_search/search_papers 依赖 MCP 服务，
       MCP 需要 GLM_CODING_PLAN_API_KEY，当前未配置

  P1 级（影响拿奖质量）：
    6. 无 HTTP API 层（只有 Python 类，评委无法调用）
    7. 训练日志输出格式与代码内置的解析器不匹配
       （training_log_parser.py 找 val_mae，但深度估计不会输出这个字段）
    8. 无独立评估脚本（计算全部 6 个深度指标并输出标准格式）
    9. max_cycles 设为 1，只跑了一轮就停了


二、修改清单（按优先级排列，每个标注 [独立]/[依赖]）
================================================================================

------------------------------------------------------------------------
[P0][独立] 修改 1：创建 NYU Depth v2 数据加载器
------------------------------------------------------------------------
现状：datasets/ 目录为空，train_dorn.py 的 import 会直接报错。

需要做的事：
在 D:\code\auto-research-agent\depth_project\datasets\ 下创建 __init__.py 和
nyu_depth_v2.py，实现 NYUDepthV2(Dataset) 类。

数据加载器要求：
  - 输入：root 目录路径 + split（train/val）
  - 目录结构假设：
    data/
      train/
        rgb/        # 0001.png, 0002.png, ...
        depth/      # 0001.png, 0002.png, ...（文件名与 rgb 一一对应）
      val/
        rgb/
        depth/
  - __getitem__ 返回 (rgb_tensor, depth_tensor)
  - RGB：读入 -> resize 到 (256, 256) 或 (480, 640) -> ToTensor -> Normalize
  - Depth：读入 -> resize 到同尺寸 -> 归一化到米 -> 返回 (1, H, W) tensor
  - 深度图格式兼容：支持 16-bit PNG（单位毫米）、.npy、.exr 三种格式
  - 深度值裁剪到 [0.5, 10.0] 米范围（NYU Depth v2 标准范围）

给 AI 的指令：
---
在 D:\code\auto-research-agent\depth_project\datasets\ 下创建两个文件：

1. __init__.py 内容：
from .nyu_depth_v2 import NYUDepthV2

2. nyu_depth_v2.py 内容要求：
- 实现 NYUDepthV2(torch.utils.data.Dataset) 类
- 构造函数：__init__(self, root, split='train', transform=None)
  目录结构：root/train/rgb/, root/train/depth/, root/val/rgb/, root/val/depth/
  自动扫描 rgb/ 目录，按文件名匹配 depth/ 下对应文件
- 支持的深度格式：.png(16bit,毫米) / .npy / .tiff / .exr，自动检测
- __getitem__ 返回 (image, depth)：
  image: float32 tensor (3, H, W)，已标准化
  depth: float32 tensor (1, H, W)，单位米，clip 到 [0.5, 10.0]
- 不指定 transform 时默认：resize 到 (256, 256)，ToTensor，Normalize
  mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
- depth 单独 resize 到 (256,256)，用最近邻插值（不引入虚假深度值）
- __len__ 返回样本数
- 加 __repr__ 方便调试
---

------------------------------------------------------------------------
[P0][依赖数据] 修改 2：准备 NYU Depth v2 数据集
------------------------------------------------------------------------
现状：data/ 目录是空的。

需要做的事：
下载 NYU Depth v2 labeled subset（官方 1449 张标注图像），
按以下结构放入 data/ 目录：

  depth_project/data/
    train/
      rgb/        # 约 1200 张（官方 train split）
      depth/      # 对应深度图
    val/
      rgb/        # 约 249 张（官方 test split）
      depth/

数据来源选项（选一个）：
  选项 A：从实验室数据集中取子集（如果实验室有 NYUv2 或类似室内深度数据）
  选项 B：下载官方 NYU Depth v2 Labeled Dataset
          https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
  选项 C：用 HuggingFace datasets 上的 nyuv2 子集
          pip install datasets; from datasets import load_dataset

如果数据是 .mat 格式（官方原始格式），需要写转换脚本：
  读 .mat -> 提取 RGB 和 depth -> 分别存为 .png

注意：训练集和验证集的图像-深度对必须文件名一一对应。

------------------------------------------------------------------------
[P0][独立] 修改 3：修复 config.yaml 指标体系
------------------------------------------------------------------------
现状：config.yaml 的 goals.metrics 写的是 val_MAE target 0.20，
但深度估计用 AbsRel/RMSE/delta1。这导致系统全程追踪错误的指标。

需要修改的文件：D:\code\auto-research-agent\depth_project\config.yaml

修改内容：
将 goals 段改为：

goals:
  metrics:
    - key: "val_AbsRel"
      target: 0.15
      direction: "lower"
    - key: "val_RMSE"
      target: 0.5
      direction: "lower"
    - key: "val_delta1"
      target: 0.75
      direction: "higher"
  stop_on_achieved: true

同时把 max_cycles 改为 -1（无限循环）或 10（至少跑 3-5 轮）：

agent:
  max_cycles: 10

------------------------------------------------------------------------
[P0][独立] 修改 4：重写训练脚本，补全评估指标 + 标准化日志输出
------------------------------------------------------------------------
现状：scripts/train_dorn.py 只计算 L1 loss，不计算 AbsRel/RMSE/delta1。
agent 无法从结果中判断假设是否成立。

需要修改的文件：scripts/train_dorn.py

修改要求：
1. 训练结束后，在验证集上计算全部深度指标：
   - AbsRel（绝对相对误差）
   - RMSE（均方根误差）
   - MAE（平均绝对误差，log 空间）
   - delta1（阈值 1.25 的精度）
   - delta2（阈值 1.25^2 的精度）
   - delta3（阈值 1.25^3 的精度）

2. 关键：日志输出格式必须被系统解析器识别。
   在每个验证 epoch 结束后，训练日志必须打印如下格式的行：

   FINAL METRICS: val_AbsRel=0.158 val_RMSE=0.52 val_MAE=0.21 val_delta1=0.78 val_delta2=0.92 val_delta3=0.97

   以及：
   Best val_AbsRel=0.158 at epoch 5

   这个格式能被 core/training_log_parser.py 和 core/fact_scanner.py 自动提取。

3. 训练参数从命令行或 config 文件读取，不硬编码。
4. 训练结束后保存 best_model.pth 到 model_snapshots/ 目录。
5. 保存训练日志到 logs/ 目录。

给 AI 的指令：
---
重写 D:\code\auto-research-agent\depth_project\scripts\train_dorn.py，要求：

1. 命令行参数：--model（模型脚本路径）, --data_dir, --batch_size,
   --epochs, --lr, --gpu, --config（可选 yaml 配置文件）
2. 从指定的 model 脚本动态 import 模型类（支持 agent 后续创建新模型文件）
3. 训练用 L1 loss 或 SILog loss（尺度不变对数损失）
4. 每 5 个 epoch 在验证集上评估，计算：
   AbsRel, RMSE, MAE(log10), delta1(1.25), delta2(1.25^2), delta3(1.25^3)
   深度评估时 clip 预测值到 [0.5, 10.0]，与真值一致
5. 日志输出格式（必须严格匹配）：
   每个 eval epoch 后打印一行：
   EVAL epoch=5 val_AbsRel=0.158 val_RMSE=0.52 val_MAE=0.21 val_delta1=0.78 val_delta2=0.92 val_delta3=0.97
   训练结束打印一行：
   FINAL METRICS: val_AbsRel=0.158 val_RMSE=0.52 val_MAE=0.21 val_delta1=0.78 val_delta2=0.92 val_delta3=0.97
   最优结果后打印：
   Best val_AbsRel=0.158 at epoch 5
6. 保存 best_model.pth 到 model_snapshots/ 目录
7. 完整训练日志写到 logs/train_<timestamp>.log
8. 数据加载用 datasets.nyu_depth_v2.NYUDepthV2
---

------------------------------------------------------------------------
[P0][独立] 修改 5：创建独立评估脚本
------------------------------------------------------------------------
现状：没有独立评估脚本，无法在训练后单独评估一个 checkpoint。

需要创建的文件：scripts/evaluate.py

给 AI 的指令：
---
创建 D:\code\auto-research-agent\depth_project\scripts\evaluate.py，要求：

1. 命令行参数：--model, --checkpoint, --data_dir, --gpu
2. 加载模型和 checkpoint
3. 在验证集上计算全部 6 个深度指标（AbsRel/RMSE/MAE/delta1/delta2/delta3）
4. 输出格式（打印到 stdout，方便 agent 解析）：
   EVALUATION RESULT:
   val_AbsRel=0.142
   val_RMSE=0.48
   val_MAE=0.19
   val_delta1=0.82
   val_delta2=0.93
   val_delta3=0.97
5. 同时保存预测深度图的可视化（彩色化）到 output/ 目录
   （供 agent 的视觉分析模块使用）
6. 按场景分桶输出指标（室内/边缘/远距离），方便定位改进方向
---

------------------------------------------------------------------------
[P0][独立] 修改 6：修复文献搜索（MCP 或替代方案）
------------------------------------------------------------------------
现状：web_search/search_papers 全部失败。
原因：MCP 服务需要 GLM_CODING_PLAN_API_KEY，当前未配置。
日志原文："All web and paper search attempts have failed"

这导致 agent 无法做真正的文献调研，只能靠模型内部知识。
对比赛叙事是硬伤——"假设有证据支撑"是评分维度之一。

解决方案（三选一，推荐方案 A）：

方案 A：配置 MCP 搜索服务 [推荐]
  - 设置 GLM_CODING_PLAN_API_KEY 环境变量（智谱 API key）
  - 或配置 Tavily/Serper/Brave Search 等搜索 API 作为 MCP 后端
  - 这样 web_search 和 search_papers 工具就能真正返回论文结果

方案 B：内置文献知识库 [备选]
  - 预下载一批深度估计领域的经典论文元数据（标题/作者/年份/核心结论/性能指标）
  - 存为 JSON 文件（如 scripts/depth_papers_db.json）
  - 修改 search_papers 工具优先查本地库，MCP 作为补充
  - 优点：不依赖网络，100% 可用
  - 缺点：覆盖面有限

方案 C：用 Qwen API 自带的联网搜索能力
  - 检查 DashScope 是否支持 enable_search 参数
  - 如果支持，在 researcher agent 的调用中开启联网搜索
  - 这样不依赖 MCP，直接用 Qwen 自带搜索

推荐方案 A + B 组合：配好 MCP 搜索 + 预置本地论文库作为保底。

------------------------------------------------------------------------
[P1][独立] 修改 7：创建 FastAPI HTTP 服务
------------------------------------------------------------------------
现状：只有 api.py（Python 类），评委无法通过 HTTP 调用。

需要创建的文件：D:\code\auto-research-agent\web\app.py（FastAPI 服务）

这是前端 Phase B 的前提条件。

给 AI 的指令：
---
在 D:\code\auto-research-agent\ 下创建 web/ 目录，创建 web/app.py：

用 FastAPI 实现 HTTP 服务，封装 AutoResearcher 类的功能。
需要安装：pip install fastapi uvicorn

端点列表：
- POST /api/project  -> 创建项目（接收 name/brief/target_metrics/dataset_path，
  生成 project 目录 + PROJECT_BRIEF.md + config.yaml，返回 project_id）
- POST /api/cycle    -> 触发一轮闭环（接收 project_id，调用 run_one_cycle，返回结果）
- GET  /api/status/{project_id}  -> 返回当前轮次/阶段/agent状态/指标/训练进度
- GET  /api/history/{project_id} -> 从 experiment_history.db 读取所有轮次历史
- GET  /api/results/{project_id} -> 返回最新指标 + 研究报告文本
- GET  /api/agents/trace/{project_id} -> 返回最近 N 条 ToolTrace 记录
- GET  /api/verify/{project_id} -> 返回 12 层验证结果 + 方法论闸门状态
- WS   /ws/status/{project_id}  -> WebSocket，每 3 秒推送 status

数据结构参考 D:\code\V-SciAgent前端v1\FRONTEND_BUILD_SPEC.md 第三节。

加 API key 鉴权（header: X-API-Key）。
加 CORS（允许前端 localhost:3000 访问）。
启动方式：uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
---

------------------------------------------------------------------------
[P1][独立] 修改 8：清理根目录临时文件
------------------------------------------------------------------------
现状：D:\code\ 下有大量 _patch_*.py、_chk_*.py、_verify.py 等临时脚本。
这些文件会干扰评委对代码质量的判断。

需要做的事：
将以下文件移到 tmp/ 或删除（确认无用后）：
  _chk_log.py, _chk_ws.py, _extract_proof.py, _patch_agents.py, _patch_api.py,
  _patch_config.py, _patch_counter.py, _patch_enc.py, _patch_mem.py,
  _patch_open.py, _patch_safe.py, _patch_tol.py, _patch_wt.py, _patch_wt2.py,
  _proof_v.py, _reset.py, _reset_state.py, _scan_open.py, _verify.py,
  gen_mini_dataset.py

注意：不要删 auto-research-agent 目录内的文件，只清理 D:\code\ 根目录。

------------------------------------------------------------------------
[P2][独立] 修改 9：整理 Qwen 合规凭证
------------------------------------------------------------------------
现状：QWEN_CALL_PROOF.txt 已有 DashScope 调用日志，但不够规范。

需要做的事：
创建 docs/qwen_compliance.md，包含：
  - 基座模型说明（Qwen-plus + Qwen-turbo，通过阿里云百炼 DashScope 调用）
  - 百炼控制台截图位置（截图保存到 docs/screenshots/）
  - 调用日志摘录（从 autoresearcher.log 提取关键行）
  - API 端点说明（https://dashscope.aliyuncs.com/compatible-mode/v1）

------------------------------------------------------------------------
[P2][独立] 修改 10：编写一键复现脚本
------------------------------------------------------------------------
需要创建的文件：D:\code\auto-research-agent\run_demo.sh（或 run_demo.ps1）

脚本内容：
  1. 检查 Python 版本和 GPU
  2. pip install -r requirements.txt
  3. 检查 DASHSCOPE_API_KEY 环境变量
  4. 检查数据集是否就位
  5. 运行 baseline 训练 1 轮
  6. 运行 agent 闭环 3 轮
  7. 输出结果摘要


三、操作流程（按依赖顺序）
================================================================================

------------------------------------------------------------------------
阶段 1：准备数据和代码骨架（半天）
------------------------------------------------------------------------
[P0][独立]  修改 1：创建 NYU Depth v2 数据加载器
[P0][依赖]  修改 2：准备 NYU Depth v2 数据集（需要你下载数据或从实验室取）
[P0][独立]  修改 3：修复 config.yaml 指标体系
[P0][独立]  修改 4：重写训练脚本
[P0][独立]  修改 5：创建评估脚本

验证点：
  cd D:\code\auto-research-agent\depth_project
  python scripts\train_dorn.py --data_dir data --epochs 5 --batch_size 4
  能跑通训练，输出 val_AbsRel 等指标，保存 best_model.pth

------------------------------------------------------------------------
阶段 2：修复文献搜索 + 跑通闭环（1-2 天）
------------------------------------------------------------------------
[P0][独立]  修改 6：修复文献搜索（配 MCP key 或建本地论文库）

验证点：
  手动跑一轮 agent 闭环：
  python -c "from api import AutoResearcher; r = AutoResearcher('depth_project'); r.run_one_cycle()"
  日志中 web_search/search_papers 能返回真实论文结果（不是 "all failed"）

然后跑 3 轮闭环：
  config.yaml 的 max_cycles 设为 3
  python -c "from api import AutoResearcher; r = AutoResearcher('depth_project'); r.run_n_cycles(3)"
  检查每轮都有：假设 -> 文献依据 -> 实验 -> 指标 -> 反思 -> 下一轮方向
  检查指标是否逐步改善

------------------------------------------------------------------------
阶段 3：搭建 HTTP API（1-2 天，可与阶段 2 并行）
------------------------------------------------------------------------
[P1][独立]  修改 7：创建 FastAPI HTTP 服务

验证点：
  uvicorn web.app:app --port 8000
  curl http://localhost:8000/api/status/depth_project 能返回 JSON
  curl http://localhost:8000/docs 能看到 Swagger 文档

------------------------------------------------------------------------
阶段 4：清理打磨（半天）
------------------------------------------------------------------------
[P1][独立]  修改 8：清理根目录临时文件
[P2][独立]  修改 9：整理 Qwen 合规凭证
[P2][独立]  修改 10：编写一键复现脚本


四、依赖关系总览
================================================================================

  [独立] 修改 1 数据加载器 ----+
                              |
  [依赖数据] 修改 2 数据集 ---+---> 验证训练能跑通
                              |         |
  [独立] 修改 3 config ------+         |
                                         |
  [独立] 修改 4 训练脚本 ----------------+
                                         |
  [独立] 修改 5 评估脚本 ----------------+
                                         |
                                         v
  [独立] 修改 6 文献搜索 ---------> 跑通 3 轮闭环
                                         |
  [独立] 修改 7 FastAPI -----------------+---------> 前端 Phase B 可开始
                                         |
  [独立] 修改 8-10 清理打磨 -------------+


五、数据集需求清单（给你的）
================================================================================

你需要准备以下数据：

1. NYU Depth v2 数据集（必需）
   - 官方 labeled subset：约 1449 张图像（1200 训练 + 249 测试）
   - 格式：RGB 图像 + 对应深度真值
   - 深度单位：米（或毫米，代码会自适应）
   - 分辨率：任意（代码会 resize 到 256x256）
   - 放到 depth_project/data/train/rgb, depth_project/data/train/depth,
     depth_project/data/val/rgb, depth_project/data/val/depth

   如果数据是 .mat 格式，告诉我，我写转换脚本。
   如果你用实验室自己的室内深度数据集也行，只要是 RGB+depth 配对。

2. DASHSCOPE_API_KEY（阿里云百炼 API key）
   - 确认环境变量已设置
   - 验证方式：echo $env:DASHSCOPE_API_KEY（PowerShell）

3. GLM_CODING_PLAN_API_KEY（智谱 API key，用于 MCP 文献搜索）[推荐配置]
   - 如果有，文献搜索就能用
   - 如果没有，需要走修改 6 的方案 B（本地论文库）


六、验收标准
================================================================================

阶段 1 验收：
  - python scripts\train_dorn.py 能跑通，输出 val_AbsRel/RMSE/delta1 等指标
  - python scripts\evaluate.py 能独立评估一个 checkpoint

阶段 2 验收：
  - agent 跑 3 轮闭环，每轮有完整的假设-实验-验证-反思记录
  - web_search/search_papers 能返回真实论文（不是 "all failed"）
  - 指标逐步改善（AbsRel 逐轮下降或 delta1 逐轮上升）
  - experiment_history.db 有 3 轮记录

阶段 3 验收：
  - FastAPI 服务可访问，Swagger 文档可打开
  - curl 能拿到真实数据
  - 前端能对接（通知前端同学开始 Phase B）

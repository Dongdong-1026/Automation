# 恒生指数早间自动化推送至 Google Chat — 设计规范

**日期**：2026-07-21
**作者**：Claude（与用户协作产出）
**状态**：待用户复核

## 1. 目标与背景

每天 HKT/北京时间 08:00（UTC 00:00）自动运行恒生指数（`^HSI`）的 LSTM 推理流程，将本次产出的预测图、关键数值、最近一日命中摘要、运行状态以一条 Google Chat 卡片消息推送到名为 **Automation Space** 的 Chat Space。

### 1.1 业务动机
- 用户早上打开 Chat 即可看到模型对当日及未来 30 个交易日的方向/收益/波动率判断，无需手动跑 Notebook
- 通过"每天必响一次"的消息建立对自动化系统的健康感
- 复用现有 LSTM 项目结构和已有 GitHub Actions 工作流模式

### 1.2 范围
**包含**：
- 一个新的 GH Actions 工作流 `daily-morning-push.yml`
- 两个新 Python 脚本：`collect_run_artifacts.py`、`push_to_google_chat.py`
- `.gitattributes` 增加 PNG LFS 规则
- 一份人工配置手册 `docs/google-chat-webhook-setup.md`
- 基础 pytest 测试

**不包含**：
- 训练模型 / 调参 / 特征筛选（已有 `monthly-hpo.yml`、`weekly-retrain.yml` 负责）
- 复盘逻辑改动（`step8_review.py` / `daily_1d_review.py` 保持原样，仅被调用）
- 多个 ticker 并行推送（本期只做 `^HSI`；如未来要扩展到 `^GSPC`、`GC=F`，重复加 matrix 项即可）
- 真正的用户身份（OAuth）发送 — 全程用 Incoming Webhook，**不**使用用户密码

## 2. 架构总览

```
┌─────────────────────────────────────────────────────┐
│ GH Actions cron: 0 0 * * 1-5 (UTC 00:00 = HKT 08:00)│
└─────────────────────┬───────────────────────────────┘
                      │
        ┌─────────────▼──────────────┐
        │ Job: daily-morning-push    │
        │ Matrix: ticker=^HSI        │
        └─────────────┬──────────────┘
                      │
       ┌──────────────┼──────────────────┐
       ▼              ▼                  ▼
   Step A          Step B             Step C
   准备环境        跑推理              推送 Chat
       │              │                  │
       │      失败→bootstrap TRAIN       │
       │              │                  │
       └──────────────┼──────────────────┘
                      ▼
              commit 产物到 repo
              (model_artifacts/latest/HSI/)
                      │
                      ▼
              collect_run_artifacts.py
              → summary.json
                      │
                      ▼
              push_to_google_chat.py
              POST webhook URL
```

## 3. 文件清单

| 路径 | 状态 | 用途 |
|---|---|---|
| `.github/workflows/daily-morning-push.yml` | 新增 | GH Actions 入口 |
| `scripts/collect_run_artifacts.py` | 新增 | 解析 run 目录产物为 JSON |
| `scripts/push_to_google_chat.py` | 新增 | 组装 + POST Google Chat 卡片 |
| `scripts/__init__.py` | 新增 | 让 scripts 成为 Python 包（便于测试导入） |
| `tests/test_collect_run_artifacts.py` | 新增 | 产物摘要器单元测试 |
| `tests/test_push_to_google_chat.py` | 新增 | Chat 推送器单元测试（含 HTTP mock） |
| `tests/conftest.py` | 新增 | pytest 共享 fixture（假 run 目录） |
| `.gitattributes` | 修改 | 补 PNG LFS 规则 |
| `docs/google-chat-webhook-setup.md` | 新增 | 用户一次性配置手册 |
| `requirements-dev.txt` | 新增 | pytest + responses 等测试依赖 |

不动：
- `LSTM_twotarget_v3.ipynb`
- `step8_review.py`、`daily_1d_review.py`
- `daily-inference.yml`、`monthly-hpo.yml`、`weekly-retrain.yml`
- `model_artifacts/` 已有内容

## 4. 组件契约

### 4.1 `collect_run_artifacts.py`

**职责**：从一次 INFERENCE 的 run 目录抽取本卡片所需的全部信息。

**输入参数**：
- `--run-dir PATH`（必填）：本次 run 目录，例如 `model_artifacts/HSI/2026-07-21/run_1400`
- `--root PATH`（必填）：artifacts 根目录，用于定位 `Step8_Review_Daily_1D.csv`
- `--ticker STR`（必填）：`^HSI`
- `--commit-sha STR`（可选）：本次 commit SHA，用于在 Chat 卡片里附 "View run" 链接

**输出（stdout，UTF-8 JSON）**：
```json
{
  "schema_version": 1,
  "ticker": "^HSI",
  "run_dir": "model_artifacts/HSI/2026-07-21/run_1400",
  "status": "ok",
  "error": null,
  "predictions": {
    "1d": 0.0042,
    "5d": 0.0118,
    "10d": 0.0203,
    "15d": 0.0276,
    "20d": 0.0321,
    "25d": 0.0355,
    "30d": 0.0389
  },
  "volatility": 0.0182,
  "direction": "up",
  "png_files": [
    {"name": "pred_path.png", "url": "https://raw.githubusercontent.com/<owner>/<repo>/<branch>/model_artifacts/latest/HSI/pred_path.png"},
    {"name": "quantile_band.png", "url": "https://raw.githubusercontent.com/<owner>/<repo>/<branch>/model_artifacts/latest/HSI/quantile_band.png"}
  ],
  "latest_1d_review": {
    "date": "2026-07-18",
    "sample_count": 12,
    "direction_accuracy_pct": 58.33,
    "pred_avg_return_pct": 0.21,
    "actual_avg_return_pct": 0.18
  },
  "commit_sha": "abc1234",
  "generated_at": "2026-07-21T08:01:42Z"
}
```

**status 枚举**：
- `ok`：INFERENCE 直接成功
- `bootstrapped`：INFERENCE 失败，bootstrap TRAIN 后重试成功
- `failed`：彻底失败（INFERENCE 和 bootstrap 都失败，或 run 目录不存在）
- `incomplete`：run 目录存在但关键产物缺失（如无 PNG、无 csv）

**错误容忍**：任何子步骤失败 → 对应字段填 `null`，status 视情况设为 `failed` 或 `incomplete`，**不抛异常**。脚本必须总能用空 summary JSON 完成，方便 Chat 推送。

**`png_files[].name` 约定**：run 目录内**相对文件名**（如 `pred_path.png`）。URL 由调用方按固定 raw 模板拼接：`https://raw.githubusercontent.com/<owner>/<repo>/<branch>/model_artifacts/latest/HSI/{name}`。脚本只负责挑出文件名，URL 拼接由 `push_to_google_chat.py` 在收到 payload 后完成（或在本脚本里通过 `--repo-owner` / `--repo-name` / `--branch` 参数拼接，二者选一——实现阶段定）。

### 4.2 `push_to_google_chat.py`

**职责**：从 summary JSON 组装 Google Chat 卡片，POST 到 webhook。

**输入参数**：
- `--payload PATH`（必填）：summary JSON 文件路径
- `--webhook-env NAME`（默认 `GOOGLE_CHAT_WEBHOOK`）：从哪个环境变量读 webhook URL

**输出**：HTTP 状态码到 stdout；非 2xx 写 stderr 并以 exit code 1 退出（由 GH Actions 重试步骤处理）。

**卡片结构**（使用 Google Chat `cardsV2`）：
- `card.header`：标题含 ticker + 日期 + 状态徽章 emoji
- `card.sections[0]`：**本次关键数值** — 表格列出 1d/5d/.../30d 预测收益 + 未来波动率 + 方向
- `card.sections[1]`：**最近 1D 命中摘要**（如 `latest_1d_review` 非空）
- `card.sections[2]`：**预测图列表** — `widgets` 含 `Image` 组件，每张 PNG 一行，`imageUrl` 为 raw.githubusercontent.com URL
- `card.sections[3]`：**Footer** — "View Actions run" 按钮 + "View latest commit" 按钮

**失败处理**：
- 缺图：跳过 `Image` widget，附文本 "（图片未生成）"
- status == failed：Header 加 ❌，sections[0] 替换为错误摘要文本
- HTTP 失败：返回非 0 退出码

### 4.3 `daily-morning-push.yml`

**触发**：
```yaml
on:
  schedule:
    - cron: "0 0 * * 1-5"   # UTC 00:00 = HKT/北京 08:00 工作日
  workflow_dispatch:        # 手动触发
```

**超时**：90 分钟（INFERENCE 60min + bootstrap TRAIN 60min + 收尾，按串行总和上限估）

**Steps**（与现有 `daily-inference.yml` 复用模式）：

1. Checkout（LFS）
2. Cache + 安装 TA-Lib
3. Setup Python 3.10
4. Cache + 安装 pip 依赖（含 `papermill`、`pytest`、`responses`）
5. `set -e; papermill ... INFERENCE` 包装在 if 里；失败则 bootstrap TRAIN + 再 INFERENCE
6. **新增**：复制本次 run 目录的 PNG/CSV/JSON 到 `model_artifacts/latest/HSI/`
7. `git add model_artifacts/latest/HSI/ && git commit && git push`（失败 continue）
8. **新增**：`python scripts/step8_review.py --root model_artifacts --ticker ^HSI`
9. **新增**：`python scripts/collect_run_artifacts.py ... > summary.json`
10. **新增**：`python scripts/push_to_google_chat.py --payload summary.json --webhook-env GOOGLE_CHAT_WEBHOOK`
11. `actions/upload-artifact@v4` 上传 `summary.json` + 本次 run 目录

**env / secrets**：
- `GOOGLE_CHAT_WEBHOOK`：从 `secrets.GOOGLE_CHAT_WEBHOOK` 注入（用户手动在 GitHub Repo Settings 配置）
- `ALLTICK_API_KEY` / `ALLTICK_API_URL`：复用现有 secrets（INFERENCE 需要）

### 4.4 `.gitattributes` 修改

新增行（保留原有规则）：
```
*.png filter=lfs diff=lfs merge=lfs -text
```

## 5. 错误处理矩阵

| 步骤 | 失败模式 | 行为 | Chat 推送 |
|---|---|---|---|
| 检出 / 装包 | LFS 未水合 / pip 失败 | workflow 标红，但 Step 10 仍推送"环境准备失败"消息 | ✅ 推送 |
| INFERENCE | papermill 报错 | 进 bootstrap 路径；STATUS=bootstrapped 或 failed | ✅ 推送 |
| bootstrap TRAIN | 也失败 | STATUS=failed；error 字段填 stderr 末 2000 字符 | ✅ 推送失败消息 |
| commit/push | 网络冲突 / 权限 | 不重试，继续下游；Chat 消息附 warning | ✅ 推送（带警告） |
| collect_run_artifacts | run 目录不存在 / 解析失败 | summary 字段填空，status=incomplete/failed | ✅ 推送 |
| push_to_google_chat | HTTP 非 2xx | 重试 2 次（指数退避）；仍失败写 $GITHUB_STEP_SUMMARY | ✅（重试） |
| 图片 URL | raw.githubusercontent.com 偶发缓存 | 卡片同时附文本链接 | ✅ |

**核心原则**：Chat **每天必响一次**。这是用户唯一感知"系统活着"的信号。

## 6. 测试策略

| 测试 | 工具 | 覆盖 |
|---|---|---|
| `tests/test_collect_run_artifacts.py` | pytest | 假 run 目录 fixture → 断言 JSON schema 字段齐全 |
| `tests/test_push_to_google_chat.py` | pytest + `responses` | 假 payload → 断言 POST 体含 `cardsV2` / header / sections |
| `tests/test_gitattributes_lfs.py` | shell in pytest | `git check-attr filter model_artifacts/latest/HSI/*.png` → 断言 `lfs` |
| workflow lint | `actionlint` | GH Actions YAML 静态检查 |
| INFERENCE smoke | workflow 自身 | papermill 失败即非 0 |

不做：端到端 Chat 推送集成测试（真出问题时 GH Actions 日志足够诊断）、模型数值正确性测试（不在本期范围）。

## 7. 一次性人工配置

### 7.1 创建 Chat Webhook
1. 打开 Google Chat → 进入 "Automation" Space
2. Space 标题旁 ▼ → Apps & integrations → Manage webhooks（或 Configure webhooks）
3. 配置：
   - Name：`HSI Morning Push Bot`
   - Avatar URL：（可选）
   - **Webhook URL**：复制
4. 不勾选"Anyone in this Space can post using this webhook"（保持只读）

### 7.2 写入 GitHub Secrets
1. 打开 GitHub Repo → Settings → Secrets and variables → Actions
2. New repository secret：
   - Name: `GOOGLE_CHAT_WEBHOOK`
   - Secret: 粘贴刚才复制的 URL

### 7.3 一次性启用（首次运行）
- 在 Actions 页面手动触发一次 `Daily Morning Push` workflow（`workflow_dispatch`）
- 验证：
  1. workflow 绿色 ✅
  2. Chat 收到一条卡片消息
  3. repo 出现 commit `[Auto] Morning push ...`
  4. `model_artifacts/latest/HSI/` 有 PNG 文件

## 8. 安全考虑

- **不**使用用户 Google 账号密码；只使用 Space 级 webhook URL
- webhook URL 写入 GitHub Secrets，**不**出现在日志中（push_to_google_chat.py 失败时不打印 URL）
- `collect_run_artifacts.py` 不解析任何用户敏感数据；产物里只有 ticker 和公开市场指标
- webhook URL 泄露影响：攻击者可冒充 bot 发消息到该 Space —— **仅影响 Automation Space**，不波及用户账号

## 9. 未来扩展（不在本期）

- 多 ticker matrix（`^GSPC`、`GC=F`）
- 周末/节假日跳过（增加 `exchange_calendars` 判断）
- Chat 消息加交互按钮（如 "Run HPO now" → 触发 `workflow_dispatch`）
- 推送成功/失败的 Slack 备份通道
- 历史消息存档到 BigQuery / Sheets

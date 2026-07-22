# HSI 推送改造 + 准确率分析模块 — 设计规范

**日期**: 2026-07-22
**作者**: 与用户协作产出
**状态**: 待用户审核

## 1. 目标与背景

### 1.1 业务目标
1. **Chat 卡片改版**：未来波动率移到最前面并大字突出；删 Step8_Volatility_Path.png；删括号内容和计算方法展示；所有中文改繁体
2. **未来波动率重新定义**：从「T+30 累计波动率」改为 LSTM vol_head 直接输出的「年化波动率」（类似 VIX 对 SP500 的概念）
3. **新增准确率分析模块**：对历史预测 vs 实际走势对比，给出模型准确率评估 + 单独的可视化（GitHub Pages 静态页）

### 1.2 范围
**包含**：
- Chat 卡片结构大改 + 繁体化
- 未来波动率数据流变更（用 vol_ann）
- 新增 `scripts/accuracy.py`（准确率计算 + 静态页生成）
- 新增 `model_artifacts/HSI/predictions_history.csv`（历史预测存档）
- 新增 `docs/accuracy.html` + `docs/accuracy_data.json`（GitHub Pages 静态页）
- 修改 `scripts/collect_run_artifacts.py`（多解析 pattern attention）
- 修改 `scripts/push_to_google_chat.py`（卡片新结构）
- 修改 `.github/workflows/daily-morning-push.yml`（新增 accuracy 步骤）

**不包含**：
- 重新训练模型（用现有 best_model.pth）
- 多 ticker 支持（仅 ^HSI）
- 用户界面（仅后端 + 卡片）
- 重新实现 LSTM 架构

## 2. 关键决策（已与用户确认）

| 决策项 | 选择 |
|---|---|
| 未来波动率公式 | 直接用 LSTM vol_head 原生输出（年化） |
| Top 10 因子来源 | Pattern attention（per-day，81 个 pattern） |
| 准确率指标 | 方向准：实际涨跌方向 = 预测涨跌方向 |
| 历史预测存储 | GitHub repo 存 CSV（用 LFS 追踪） |
| 6 月准确率可视化 | GitHub Pages 静态网站 |
| 繁体范围 | Card 可见文字（不影响 LLM prompt） |
| 架构方案 | 单 workflow 全做（daily-morning-push 一次性） |

## 3. 架构数据流

```
[1] 跑 LSTM (现有)
   LSTM_twotarget_v3_P5_diag.ipynb
   ↓ 输出
   - model_artifacts/HSI/{DATE}/run_{TIME}/
     ├── Proportional_Inference_Report.txt
     ├── pattern_attention_full_report.csv
     └── best_model.pth

[2] collect_run_artifacts.py (改)
   解析 → summary.json
   包含：
     - predictions: {1d, 5d, ..., 30d} (return 小数)
     - vol_ann: 14.6 (年化 %)
     - direction: "up"/"down"/"neutral"
     - pattern_attention: {name: weight, ...}  ← 新
     - png_files, commit_sha, latest_1d_review (现有)

[3] 追加到 predictions_history.csv (新)
   model_artifacts/HSI/predictions_history.csv
   字段详见第 4 段

[4] scripts/accuracy.py (新)
   a) 读 predictions_history.csv
   b) yfinance 拉最新真实数据（截止今天）
   c) 算每个 horizon 的方向准
   d) 分组：
      - 近 6 个月每月准确率
      - 最近 30 天同 target_date 多预测日对比 → 找最准
   e) 取最准那天的 top 10 pattern attention
   f) 写 docs/accuracy_data.json + docs/accuracy.html

[5] push_to_google_chat.py (大改)
   卡片新结构（详见第 5 段）+ 繁体 + 大字 vol + 删 Volatility_Path.png

[6] git commit + push
   - predictions_history.csv
   - docs/accuracy.html
   - model_artifacts/latest/HSI/ (除 Volatility_Path 外的 PNG)
```

## 4. predictions_history.csv Schema

文件路径：`model_artifacts/HSI/predictions_history.csv`（LFS 追踪）

| 字段 | 类型 | 示例 | 说明 |
|---|---|---|---|
| `prediction_date` | date | 2026-07-22 | 跑模型的日期 |
| `ticker` | str | ^HSI | |
| `T+1_pred` | float | 0.0007 | T+1 预测收益（小数）|
| `T+1_actual` | float / null | 0.0021 | 实际收益（未发生则为 null）|
| `T+1_correct` | bool / null | true | sign(pred) == sign(actual) |
| ... | ... | (重复 5d/10d/15d/20d/25d/30d) | |
| `vol_ann` | float | 14.6 | 年化波动率 % |
| `direction` | str | up | 方向判断 |
| `top1_pattern` | str | "MA5_Cross_MA20_Bullish" | top 1 pattern 名 |
| `top1_weight` | float | 0.082 | attention weight |
| ... | ... | (重复到 top10) | |

**初始化**：从现有 `run_154008/Proportional_Inference_Report.txt` 导入一行历史记录

## 5. Chat 卡片新结构（繁體）

```
┌──────────────────────────────────────────┐
│ ✨ AI 總結（如果有 LLM 配置）              │
│ 黃金文字 1-2 句                          │
├──────────────────────────────────────────┤
│                                          │
│  📊 未來波動率                              │  ← 新位置：第 1 段
│     14.60%                                │  ← 特大字（font 28px）
│                                          │
├──────────────────────────────────────────┤
│ 📈 本次關鍵預測                            │
│ • 1d: +0.08%                              │
│ • 5d: -1.48%                              │
│ ...（7 個週期）                            │
├──────────────────────────────────────────┤
│ 🎯 最近 1D 命中摘要                        │
│ 日期：2026-07-18 · 樣本：12              │
│ 方向命中率：58.33%                        │
├──────────────────────────────────────────┤
│ 🖼️ 預測圖（不含 Volatility_Path.png）    │
│  [Step8_Proportional_Price_Path.png]     │
│  [其他 PNG]                                │
├──────────────────────────────────────────┤
│ [View commit 按鈕]                         │
│ 由 daily-morning-push 自動生成              │
└──────────────────────────────────────────┘
```

**变化点**：
1. 「未來波動率」段提到第 1 位
2. 大字显示（font 28px，weight bold）
3. 改用 `summary["vol_ann"]`（LSTM vol_head 原生输出）
4. 删掉所有 `(T+30 日級)` 括号标签
5. 删掉「└ 計算：LSTM vol_head → 日級 σ × √30」计算行
6. **所有中文改繁體**（未來/本次/關鍵/預測/方向/判斷/波動率/累計 等）
7. **不 push** `Step8_Volatility_Path.png`
8. 删除「└ T+30 累計波動率參考」参考行

## 6. 准确率计算（scripts/accuracy.py）

### 6.1 方向准定义
```
direction_correct = sign(pred) == sign(actual) AND pred != 0 AND actual != 0
accuracy = sum(direction_correct) / sum(非空样本)
```

### 6.2 月度准确率（近 6 月）
- 输入：CSV 全部行
- 分组：`groupby(prediction_date.to_period('M'))`
- 输出：每个月的方向准（7 周期平均）
- 仅显示有 ≥ 5 个样本的月份

### 6.3 最近 30 天同 target_date 找最准
- 输入：CSV 里 `prediction_date` 在最近 30 天内的行
- 对每个 target_date（=prediction_date + horizon 天）：
  - 找所有 `prediction_date + horizon == target_date` 的行
  - 算每个 pred 的 |pred - actual|
  - 误差最小者 = 最准
- 输出：表格（target_date, best_pred_date, horizon, pred, actual, error）

### 6.4 Top 10 因子
- 输入：6.3 找到的最准那一天的 CSV 行
- 从该行的 `top1_pattern` 到 `top10_pattern` / `_weight` 读出
- 加上 attention attention 本身（可选：用 pattern_attention_full_report.csv 拿到精确值）

### 6.5 节假日 / 未来数据处理
- yfinance 返回 NaN → `T+h_actual = null` → 不计入准确率分母
- 未来日期 → `T+h_actual = null` → 同上
- 跳过 df.isnull() 的行

## 7. GitHub Pages 静态页（docs/accuracy.html）

**入口**：`https://dongdong-1026.github.io/Automation/accuracy.html`

**页面结构**：
```
[標題] HSI 預測準確率分析（更新於 2026-07-22）

[區塊 1] 整體概覽
  - 整體方向命中率：X%（過去 6 個月平均）
  - 總樣本數：N
  - 最後更新：YYYY-MM-DD

[區塊 2] 近 6 個月每月準確率（條形圖）
  2026-02: 55%  ████████░░░░
  2026-03: 60%  █████████░░░
  ...

[區塊 3] 最近 30 天同目標日期對比
  表格：
  目標日期  | 最佳預測日  | horizon | 預測值 | 實際值 | 誤差
  2026-07-20 | 2026-07-15  | T+5     | +0.4%  | +0.5%  | 0.1% ← 最佳

[區塊 4] 最準那天的 Top 10 因子
  ▓▓▓▓▓▓▓▓▓▓ MA5_Cross_MA20_Bullish     8.2%
  ▓▓▓▓▓▓▓▓▓  Realized_Vol_GK            7.5%
  ▓▓▓▓▓▓▓▓   Bollinger_Band_Compression 6.8%
  ...
```

**HTML 实现要求**：
- 单文件，内嵌 CSS + JS（无外部依赖）
- 图表用原生 SVG（避免 Chart.js 等库）
- 颜色用 GitHub Primer palette
- 响应式（移动端可读）
- 数据从 `accuracy_data.json` 读（同目录）

## 8. 错误处理矩阵

| 步骤 | 失败模式 | 行为 | Chat 推送 |
|---|---|---|---|
| yfinance 拉真实数据 | 网络错误 / 限流 | 跳过缺失 actuals（null），不算准确率 | ✅ 推送（无历史对比）|
| CSV 读 / 写 | 文件不存在（首次） | 自动初始化空 CSV | ✅ 推送 |
| 静态页生成 | Jinja2 模板错误 | 写入空页 + stderr 警告 | ✅ 推送 |
| 找不到 top 10 pattern | pattern_attention 解析失败 | 跳过 top 10 段，其他照常 | ✅ 推送 |
| yfinance 节假日 | 缺失日期 | 标记 actual=null，不计分母 | - |

**核心原则**：准确率模块的所有失败都不阻塞 Chat 推送（推送是核心交付物）。

## 9. 测试策略

| 测试 | 工具 | 覆盖 |
|---|---|---|
| `tests/test_accuracy.py` | pytest | _compute_direction_accuracy, _parse_csv, 节假日处理 |
| `tests/test_collect_run_artifacts.py` | pytest | vol_ann parsing, pattern_attention parsing |
| `tests/test_push_to_google_chat.py` | pytest | 卡片新结构（无 Vol_Path PNG、大字 vol、繁体）|
| `tests/test_html_generator.py` | pytest | accuracy_data.json 结构、HTML 包含关键 section |

## 10. 未来扩展（不在本期）

- 多 ticker 准确率对比页
- 模型 A/B 测试对比
- 历史回测曲线（实际 vs 预测随时间）
- 交互式时间范围选择器
- 自动告警（准确率 < 阈值时通知）

## 11. 安全考虑

- 静态页只展示公开数据（无 secret）
- predictions_history.csv 不含敏感信息
- yfinance 公共 API（无 auth）
- GitHub Pages 自动 https
- 旧 webhook 凭据已轮换（issue 跟踪在项目记忆中）

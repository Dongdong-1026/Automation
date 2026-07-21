# Google Chat Webhook 一次性配置手册

按本手册完成 webhook 创建与 GitHub Secrets 配置，本项目每天早上 8:00（HKT/北京时间）就能自动向 Automation Space 推送模型预测卡片。

## 步骤 1：在 Google Chat 创建 Incoming Webhook

1. 打开 Google Chat（Web 或桌面客户端）
2. 进入名为 **Automation** 的 Space
3. Space 标题旁的下拉箭头 → **Manage webhooks**（若显示为 "Configure webhooks" 同义）
4. 在 "Incoming webhooks" 区域点击 **Add another webhook** 或编辑现有 webhook：
   - **Name**: `HSI Morning Push Bot`
   - **Avatar URL**: （可留空）
   - **Webhook URL**: 复制生成的 URL
5. **不要**勾选 "Anyone in this Space can post using this webhook"——保持 bot 权限最小
6. 保存

> ⚠️ 该 URL 视为半秘密：任何持有者都可以"以 bot 名义"在该 Space 发消息。请勿外发。

## 步骤 2：在 GitHub 仓库配置 Secret

1. 打开本仓库 → **Settings** → **Secrets and variables** → **Actions**
2. 点击 **New repository secret**
3. 填写：
   - **Name**: `GOOGLE_CHAT_WEBHOOK`
   - **Secret**: 粘贴步骤 1 复制的 URL
4. 点击 **Add secret**

## 步骤 3：首次手动触发验证

1. 打开仓库 → **Actions** → **Daily Morning Push (HSI → Google Chat)**
2. 点击 **Run workflow** → **Run workflow**（绿色按钮）
3. 观察 workflow 进度：
   - 全部步骤绿色 ✅
   - 在 Automation Space 收到一条卡片消息
   - 仓库出现一条 commit：`[Auto] Morning push YYYY-MM-DD ^HSI`
   - `model_artifacts/latest/HSI/` 目录下出现 PNG 文件

## 步骤 4：验证定时调度

工作日早 8:00（HKT）后 1-2 分钟内：
- Actions 页面应出现一条新的 "Daily Morning Push" run
- Automation Space 收到卡片

## 故障排查

| 现象 | 可能原因 | 解决 |
|---|---|---|
| Workflow 失败：`LFS files are still pointers` | LFS 未水合 | 检查 repo 容量、`.gitattributes` 是否含 `*.png filter=lfs` |
| Workflow 失败：`papermill` 报错 | INFERENCE 异常 | 查看 `output_inference_*.ipynb` cell outputs |
| Chat 未收到消息 | Webhook URL 错 / Secret 未注入 | 重新执行步骤 1-2；手动触发一次 |
| Chat 收到但无图 | PNG 提交失败 | 看 Actions 日志的 "Commit latest artifacts" 步骤 |
| Chat 收到但显示 "图片未生成" | `model_artifacts/latest/HSI/` 为空 | 看 INFERENCE run 目录有无 PNG；可能是模型未生成图 |

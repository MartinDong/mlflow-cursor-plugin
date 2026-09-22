# MLflow Cursor tracing

把 Cursor 每一轮 agent 上报成一条 MLflow trace（思考、工具、失败、嵌套 subagent），并用对话 id 串成 session。

## 当前机器（Windows）一键启用

在本仓库根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File install-system.ps1
```

脚本会：

1. 确保 `.venv` 已安装依赖（没有则调用 `install.ps1`）
2. 读取 `config.env`，写入用户级 `~/.cursor/hooks.json`（全局对 Cursor agent 生效）
3. 把 `mlflow-mcp` 合并进 `~/.cursor/mcp.json`（可用 MCP 查 trace）
4. 写入本仓库 `.cursor/hooks.json`（打开本项目时也生效）

然后：**Developer: Reload Window** 重载 Cursor（改 hooks 后必须重载，否则仍用旧命令）。

下一轮 Agent 对话结束后，可在 MLflow 实验里看到 trace；本地缓冲在工作区 `.cursor/mlflow/`。

> Windows 注意：hook 命令使用无空格的绝对路径直接调 `.venv\Scripts\python.exe`（不要用 `.cmd` / 带空格的 `Program Files\nodejs`）。Cursor 在 Windows 上会经 PowerShell 读 hook 的 UTF-8 临时文件再写入 stdin；系统代码页为 GBK（936）时，中文会被「UTF-8 当 ANSI 读」双重编码。插件会自动做无损还原；若仍出现 `ignored bad hook input`，请安装 **PowerShell 7**（`winget install --id Microsoft.PowerShell -e`）并保证 `pwsh` 在 PATH 上，然后 **Reload Window**。备选：系统区域设置里开启「Beta: 使用 Unicode UTF-8」。

### 配置

编辑 `config.env`（不要提交口令）：

| 变量 | 含义 |
|------|------|
| `MLFLOW_TRACKING_URI` | 跟踪服务，如 `http://127.0.0.1:21103` |
| `MLFLOW_EXPERIMENT_ID` | 实验 id |
| `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` | Basic Auth |

改完 `config.env` 后重新跑一次 `install-system.ps1`，以刷新 MCP 里的 URI。同名环境变量会覆盖 `config.env`。

### 自检

```powershell
# hook 入口（应打印 {} 或 {"continue": true}，并写入 .cursor/mlflow）
$env:CURSOR_PROJECT_DIR = (Get-Location).Path
'{"hook_event_name":"beforeSubmitPrompt","conversation_id":"smoke-1","prompt":"hi"}' |
  & .\.venv\Scripts\python.exe .\scripts\mlflow_cursor.py

# 看最近日志
Get-Content .cursor\mlflow\cursor_tracing.log -Tail 20
```

日志里出现 `hook beforeSubmitPrompt` / `hook stop` 且有 `tr-...` trace id 即表示成功。若 prompt 仍是乱码或只有 `ignored bad hook input`：确认已装 `pwsh`、重跑 `install-system.ps1` 并 **Reload Window**；也可看 `.cursor/mlflow/cursor_tracing.log` 是否有 `recovered prompt from transcript_path`。

编码自检：

```powershell
.\.venv\Scripts\python.exe tests\test_encoding_repair.py
```

## 装到其他环境

### 1. 拷贝目录

把整个 `cursor-plugin` 拷到目标机器，或放到目标仓库的：

- `.cursor-plugin/mlflow-cursor-tracing`（作为项目插件），或
- 任意子目录（再靠 hooks / MCP 手动挂载）

### 2. 安装依赖

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File install.ps1
```

```bash
# Linux / macOS / WSL
bash install.sh
```

WSL 下项目在 `/mnt/c`、`/mnt/f` 等 Windows 盘时，`install.sh` 会优先用 Windows Python 建 venv，便于 Cursor Desktop 调用。

### 3. Windows 系统启用（推荐）

```powershell
powershell -ExecutionPolicy Bypass -File install-system.ps1
```

### 4. 仅项目内启用（不写用户全局配置）

复制 hooks 到项目：

```json
{
  "version": 1,
  "hooks": {
    "beforeSubmitPrompt": [{ "command": "node path/to/cursor-plugin/scripts/run_hook.js", "timeout": 8 }],
    "stop": [{ "command": "node path/to/cursor-plugin/scripts/run_hook.js", "timeout": 90 }],
    "sessionEnd": [{ "command": "node path/to/cursor-plugin/scripts/run_hook.js", "timeout": 90 }]
  }
}
```

完整事件列表见 `hooks/hooks.json`。需要本机有 **Node.js**（Cursor 常见环境都有）和已创建的 `.venv`。

或作为 Cursor 项目插件：目录内保留 `.cursor-plugin/plugin.json`，hooks 使用 `${CURSOR_PLUGIN_ROOT}`（已在仓库 `hooks/hooks.json` 配好）。

### 5. MCP（查询 trace）

需要本机已安装 `uv`。`install-system.ps1` 会按 `config.env` 写入用户 `mcp.json`。

手动合并时，把 `mcp.json` 里的 `mlflow-mcp` 段拷进 `~/.cursor/mcp.json` 或项目 `.cursor/mcp.json`，并换成你的 URI / 实验 id。

## 行为说明

- 事件先落盘到工作区 `.cursor/mlflow/`，一轮结束再上传。
- `stop` 与 `sessionEnd` 共用锁，同一缓冲不会传两次；只有 prompt、尚无 agent 步骤时会推迟上传。
- 同一段思考、同一个 `tool_use_id` 只保留一条。
- 用户问题与回复在 root span 的 `messages` 里，避免 MLflow 3.16 详情页重复绘制。
- 运行时需要 Python 3.10+（由 `.venv` 提供）。依赖见 `requirements.txt`（MLflow 3.16.1）。

## 检查上传

```powershell
.\.venv\Scripts\python.exe tests\test_session_upload.py
```

```bash
.venv/bin/python tests/test_session_upload.py
```

跟踪服务须可从本机访问，且 `config.env` 密码正确。

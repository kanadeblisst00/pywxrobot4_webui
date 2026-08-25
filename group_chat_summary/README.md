# 群聊消息总结服务

这是一个可独立启动的子项目，提供：

- 群聊消息总结 HTTP API；
- 分段抽取与层级归并流水线；
- 话题、决定、待办、未决问题、风险的结构化输出；
- 每条结论对应的原始消息 ID；
- llama.cpp / GGUF 本地推理及自定义 OpenAI 兼容模型；
- 模型检测、摘要试运行、历史记录和参数设置管理页面。

它不依赖主 WebUI 的进程和数据库。运行数据默认写入 `data/summary.sqlite3`。

## 1. 快速启动

先从 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 获取适合硬件的构建，并确保 `llama-server.exe` 可以从终端执行。Windows NVIDIA 用户应选择 CUDA 构建并保留压缩包内的 DLL。

终端一：启动默认 Qwen3.5-9B Q4_K_M。首次启动会通过 llama.cpp 从 Hugging Face 获取约 5.7GB 的 GGUF；群聊总结只使用文本，因此脚本会关闭多模态投影下载。

```powershell
cd group_chat_summary
.\scripts\start_llama_server.ps1
```

如果已经下载了 GGUF：

```powershell
.\scripts\start_llama_server.ps1 `
  -ModelPath D:\models\Qwen3.5-9B-Q4_K_M.gguf
```

终端二：安装服务依赖并启动管理服务：

```powershell
cd group_chat_summary
python -m pip install -r requirements.txt
python run.py
```

然后访问：

- 管理页面：`http://127.0.0.1:28120/`
- OpenAPI 文档：`http://127.0.0.1:28120/docs`
- 健康检查：`http://127.0.0.1:28120/api/v1/health`

## 2. 内置模型

首次启动会创建以下预设，用户可在管理页面启用、编辑和切换：

| 配置 | llama.cpp alias | 默认接口 | 用途 |
| --- | --- | --- | --- |
| Qwen3.5 9B | `qwen3.5-9b` | `127.0.0.1:18080` | 默认启用，效果与占用均衡 |
| Qwen3.5 4B | `qwen3.5-4b` | `127.0.0.1:18081` | 低资源备选，默认停用 |
| Qwen3.5 27B | `qwen3.5-27b` | `127.0.0.1:18082` | 高质量备选，默认停用 |

默认脚本启动 9B 服务。需要切换模型时，使用对应 GGUF、alias 和端口启动另一个 `llama-server`，再到“模型管理”启用对应配置。原有 Ollama 配置作为兼容项保留但默认停用；也可以添加 vLLM、LM Studio 或其他提供 `/v1/chat/completions` 的服务。

llama.cpp 对 JSON Schema 使用自己的 `response_format.schema` 结构，本项目会根据模型配置的 `provider` 自动选择正确请求格式。

## 3. 提交摘要任务

```powershell
$body = @{
  room_id = "123@chatroom"
  room_name = "产品研发群"
  model_profile_id = "llamacpp-qwen35-9b"
  messages = @(
    @{
      id = "m101"
      sender_id = "wxid_a"
      sender_name = "林雨"
      timestamp = "2026-08-24T09:12:00+08:00"
      content = "登录模块联调完成，今天可以合入测试分支。"
      message_type = "text"
    },
    @{
      id = "m102"
      sender_id = "wxid_b"
      sender_name = "周远"
      timestamp = "2026-08-24T09:14:00+08:00"
      content = "我今天 18 点前提交支付幂等修复。"
      message_type = "text"
    }
  )
} | ConvertTo-Json -Depth 6

$job = Invoke-RestMethod `
  -Uri http://127.0.0.1:28120/api/v1/summaries `
  -Method Post `
  -ContentType application/json `
  -Body $body

Invoke-RestMethod "http://127.0.0.1:28120/api/v1/summaries/$($job.id)"
```

`POST /api/v1/summaries` 返回 HTTP 202。客户端通过任务 ID 查询状态：

- `pending`：等待执行；
- `running`：模型处理中；
- `completed`：结果位于 `result`；
- `failed`：错误信息位于 `error`。

### 消息字段

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `id` | 是 | 稳定且唯一的消息 ID，用于证据回链 |
| `content` | 是 | 消息文本或已转换为文本的内容 |
| `sender_name` | 否 | 发送者显示名，默认“未知成员” |
| `sender_id` | 否 | 发送者 wxid 或业务 ID |
| `timestamp` | 否 | ISO 8601 时间或普通时间文本 |
| `message_type` | 否 | 默认 `text`，可在设置中配置忽略类型 |

## 4. API 概览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/v1/summaries` | 创建摘要任务 |
| `GET` | `/api/v1/summaries/{id}` | 查询任务与结果 |
| `GET` | `/api/v1/summaries` | 查询历史 |
| `DELETE` | `/api/v1/summaries/{id}` | 删除任务与原文 |
| `GET/POST` | `/api/v1/model-profiles` | 列出/新增模型配置 |
| `PUT` | `/api/v1/model-profiles/{id}` | 更新模型配置 |
| `POST` | `/api/v1/model-profiles/{id}/test` | 检测服务与模型 |
| `GET/PUT` | `/api/v1/settings` | 读取/更新流水线参数 |

完整请求和响应结构以 `/docs` 为准。

## 5. 摘要策略

流水线不会把所有群聊一次性塞给模型：

1. 根据消息类型过滤无效内容；
2. 按字符预算切分，保留少量相邻消息作为重叠上下文；
3. 每个分段独立抽取结构化事实；
4. 摘要过多时进行多层归并；
5. 服务端删除不存在于原消息集合中的证据 ID；
6. 保存最终摘要和可选的原始消息。

默认每段 12,000 字符。该值不等同于模型 token 数，但对中英文混合群聊更容易控制，也不需要绑定某一个 tokenizer。

## 6. 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SUMMARY_HOST` | `127.0.0.1` | 监听地址 |
| `SUMMARY_PORT` | `28120` | 监听端口 |
| `SUMMARY_DATA_DIR` | `./data` | SQLite 数据目录 |
| `SUMMARY_REQUEST_TIMEOUT` | `180` | 单次模型请求超时秒数 |
| `SUMMARY_CORS_ORIGINS` | 空 | 逗号分隔的跨域来源；默认不开放跨域 |
| `LLAMA_CPP_BASE_URL` | `http://127.0.0.1:18080/v1` | 默认 llama.cpp 接口 |

llama-server 启动脚本还支持 `LLAMA_SERVER_BIN`、`LLAMA_MODEL_PATH`、`LLAMA_HF_REPO`、`LLAMA_MODEL_ALIAS`、`LLAMA_PORT`、`LLAMA_CONTEXT_SIZE`、`LLAMA_GPU_LAYERS` 和 `LLAMA_PARALLEL`。完整示例见 `.env.example`。

若开放到局域网或公网，请在服务前增加反向代理、认证和 TLS。当前版本定位为本机或可信内网服务，未内置账号系统。

## 7. Docker

```powershell
docker compose up --build
```

Compose 会同时启动 CPU 版 llama.cpp 和摘要管理服务，并把模型缓存放入 Docker volume。NVIDIA 环境可将镜像改为 `ghcr.io/ggml-org/llama.cpp:server-cuda`，同时为容器开启 GPU，并添加 `--n-gpu-layers 99`。

## 8. 测试

```powershell
python -m pip install -r requirements-dev.txt
pytest
```

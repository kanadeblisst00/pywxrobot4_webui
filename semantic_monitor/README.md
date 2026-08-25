# 群消息语义监测服务

这是一个可独立启动的 FastAPI 子项目，用于中文群消息的关键内容监测。它不依赖主 WebUI 的进程和数据库，通过 HTTP 接口接收消息，并提供 Web 管理台。

## 已实现

- 关键词、正则、排除规则的高精度命中
- 正反例驱动的语义相似度召回
- 群聊上下文、否定语境和反例抑制
- 多模型选择：内置字符模型、BGE、Qwen3 Embedding、本地 Qwen3Guard、自训练 n-gram、OpenAI 兼容服务
- 监测主题、风险等级和双阈值管理
- 告警记录、人工复核与误报反馈
- CSV / JSON / JSONL / TXT 数据集导入
- 零第三方机器学习依赖的字符 n-gram 分类器训练、评测和激活
- NudeNet 本地图片色情检测与可配置阈值
- RapidOCR 海报文字提取，并自动复用文本审核策略
- ZXing 二维码存在性检测（不读取、不返回二维码内容）
- SQLite 持久化、可选 API Token、交互式 OpenAPI 文档

默认只记录和提示风险，不执行踢人、封禁或删除消息等不可逆操作。

## 快速启动

```powershell
cd semantic_monitor
python -m pip install -r requirements.txt
python main.py
```

启用完整图片审核依赖：

```powershell
python -m pip install onnxruntime opencv-python-headless pyclipper shapely
python -m pip install --no-deps rapidocr-onnxruntime nudenet
```

第二条命令使用 `--no-deps`，是为了让 RapidOCR 和 NudeNet 共用前一步安装的
ONNX Runtime 与无界面版 OpenCV，避免重复安装互相覆盖的 OpenCV 包。

打开：

- 管理台：<http://127.0.0.1:28110/>
- API 文档：<http://127.0.0.1:28110/docs>
- 健康检查：<http://127.0.0.1:28110/api/health>

运行自动化测试：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## 接入微信机器人

主项目已提供 `plugins/semantic_monitor_guard.py`。在消息插件页面启用后，至少需要：

1. 选择需要审核的群聊；
2. 保持接口地址为 `http://127.0.0.1:28110/api/v1/messages/analyze`；
3. 如果审核服务设置了 Token，同时填写相同的 API Token；
4. 首次使用建议关闭“群内提醒”，仅观察审核中心和插件日志，校准后再开启。

审核服务故障或超时时插件会放行消息并记录警告，不会阻塞主机器人。

也可以由其他客户端直接调用接口：

机器人收到群消息后，将标准化字段 POST 到检测接口：

```http
POST /api/v1/messages/analyze
Content-Type: application/json
X-API-Token: 仅在配置了 Token 时填写
```

```json
{
  "message_id": "123456",
  "room_id": "123@chatroom",
  "sender_id": "wxid_example",
  "sender_name": "小王",
  "text": "能不能先支援两千，下周还你",
  "message_type": "text",
  "context": [
    "小李：最近大家都挺忙",
    "小王：我工资要下周才发"
  ],
  "metadata": {
    "wxpid": 10001
  },
  "persist": true
}
```

响应示例：

```json
{
  "message_id": "123456",
  "matched": true,
  "risk_level": "high",
  "processing_ms": 3.28,
  "matches": [
    {
      "topic_name": "资金借贷请求",
      "matched": true,
      "confidence": 0.91,
      "stage": "semantic",
      "evidence": "正例相似 1.00，反例相似 0.10；上下文未发现明显冲突",
      "needs_review": true
    }
  ],
  "event_ids": [1]
}
```

建议调用方根据 `matched`、`risk_level` 和 `needs_review` 生成管理员通知。高风险结果仍应人工确认。

## 模型选择

模型设置可在管理台修改，也可以调用 `GET/PUT /api/v1/models/settings`。

| 场景 | Embedding | 分类 / 复核 |
|---|---|---|
| 首次启动、普通 CPU | `builtin/char-ngram-zh-v1` | 内置规则或自训练 n-gram |
| 中文语义召回 | `BAAI/bge-small-zh-v1.5`、`BAAI/bge-m3` | 内置复核 |
| 更强中文召回 | `Qwen/Qwen3-Embedding-0.6B` | Qwen3Guard 或兼容接口 |
| 完全本地 | Sentence Transformers | Ollama / vLLM / LM Studio |
| 已有模型网关 | OpenAI 兼容 `/v1/embeddings` | OpenAI 兼容 Chat 接口 |

使用 BGE 或 Qwen3 Embedding 前安装可选依赖：

```powershell
python -m pip install -r requirements-models.txt
```

首次启用本地 Transformer 模型会下载权重；模型大小和许可证应在部署前自行确认。若群消息不能离开本机，请选择内置或本地模型，不要填写外部 API 地址。

## 关键 API

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/v1/messages/analyze` | 分析单条消息 |
| `POST` | `/api/v1/images/analyze` | 上传图片并执行色情、OCR、二维码存在性检测 |
| `GET` | `/api/v1/vision/status` | 查看图片组件开关和依赖可用状态 |
| `POST` | `/api/v1/test/batch` | 批量回归测试 |
| `GET/POST` | `/api/v1/topics` | 查询、新建监测主题 |
| `PUT/DELETE` | `/api/v1/topics/{id}` | 更新、删除主题 |
| `GET/PUT` | `/api/v1/models/settings` | 查询、保存模型选择 |
| `POST` | `/api/v1/models/probe` | 测试模型可用性 |
| `GET` | `/api/v1/events` | 查询告警和复核队列 |
| `POST` | `/api/v1/events/{id}/feedback` | 标记准确或误报 |
| `GET` | `/api/v1/datasets/catalog` | 查询内置数据集目录 |
| `POST` | `/api/v1/datasets/import` | 导入本地标注文件 |
| `POST` | `/api/v1/training/runs` | 启动轻量分类器训练 |
| `POST` | `/api/v1/models/activate/{run_id}` | 激活训练完成的模型 |

完整字段定义以 `/docs` 为准。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SEMANTIC_MONITOR_HOST` | `127.0.0.1` | 监听地址 |
| `SEMANTIC_MONITOR_PORT` | `28110` | 监听端口 |
| `SEMANTIC_MONITOR_DB` | `data/semantic_monitor.sqlite3` | SQLite 路径 |
| `SEMANTIC_MONITOR_API_TOKEN` | 空 | 留空关闭鉴权 |
| `SEMANTIC_MONITOR_CONTEXT_SIZE` | `8` | 每个群保留的上下文条数 |

生产环境若监听非本机地址，请设置高强度 `SEMANTIC_MONITOR_API_TOKEN`，并通过 HTTPS 反向代理访问。

## 目录

```text
semantic_monitor/
├── app/                 FastAPI、检测引擎、模型和数据集逻辑
├── datasets/            内置的最小演示数据
├── static/              独立 Web 管理台
├── tests/               API 与分类器测试
├── data/                运行时数据库和训练模型（不提交）
└── main.py              独立启动入口
```

内置数据只用于验证训练链路，不能替代你的真实群聊标注。正式使用时应持续收集“准确、误报、漏报”反馈，用独立测试集校准每个主题的阈值。

## 图片审核边界

- 图片只在内存或操作系统临时目录中处理，审核中心不会把上传原图保存到数据目录。
- OCR 文本会进入现有文本审核链路；选择 `persist=true` 时，命中记录会写入审核事件表。
- NudeNet 命中时只记录检测类别与置信度，不保存检测框截图。
- 二维码模块只访问条码格式并统计数量，代码中不会读取 `barcode.text` 或二维码字节内容。
- 当前插件默认不会因二维码在群内警告。若需要通知管理员，可开启“检测到二维码时通知管理员”。

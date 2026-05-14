# 智能客服系统 (Customer Service Agent)

基于 LangChain + LlamaIndex + Milvus 构建的智能客服系统，支持私有知识库问答、多模型接入、精排重排、可观测性监控。

## 快速启动

### 前置依赖

- Docker & Docker Compose
- （可选）`hf` CLI 用于预下载模型：`brew install pipx && pipx install huggingface-hub`

### 1. 配置

复制示例配置文件并修改：

```bash
cp conf/settings.example.yaml conf/settings.yaml
```

编辑 `conf/settings.yaml`，配置你的 LLM API 信息：

```yaml
llm:
  base_url: "https://api.openai.com/v1"  # 支持任何 OpenAI 兼容接口
  api_key: "your-api-key"
  model_name: "gpt-4"
```

### 2. 下载模型（推荐离线方式）

系统需要两个小模型：Embedding 模型和精排模型。首次启动会自动下载，但国内网络可能不稳定，**推荐预下载到本地**：

```bash
# 设置 HuggingFace 国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 下载 Embedding 模型（约 1.3GB）
hf download BAAI/bge-large-zh-v1.5 --local-dir ./models/bge-large-zh-v1.5

# 下载精排模型（约 1.1GB）
hf download BAAI/bge-reranker-base --local-dir ./models/bge-reranker-base
```

> **注意**：这些不是大语言模型（LLM），而是轻量的向量化/打分模型，在 CPU 上即可运行。
> LLM 调用通过远程 API 完成，不在本地运行。

下载完成后，在 `conf/settings.yaml` 中配置本地路径：

```yaml
embedding:
  model_name: "/app/models/bge-large-zh-v1.5"  # 容器内路径

reranker:
  model_name: "/app/models/bge-reranker-base"   # 容器内路径
```

如果不预下载，也可以使用 HuggingFace 模型名（需容器内网络可达）：

```yaml
embedding:
  model_name: "BAAI/bge-large-zh-v1.5"

reranker:
  model_name: "BAAI/bge-reranker-base"
```

### 3. 准备知识库文档

将你的知识库文档放入 `knowledge_docs/` 目录，支持 txt、md、pdf、docx 等格式。

### 4. 启动服务

```bash
docker compose up -d --build
```

等待所有服务启动完成后：
- API 服务: http://localhost:8000
- Grafana 监控面板: http://localhost:3000 (admin/admin)
- Attu (Milvus 管理): http://localhost:8989

> **首次构建提示**：
> - 如果 pip 安装慢，Dockerfile 已配置清华镜像源加速
> - 如果 Docker 镜像拉取慢，可配置 Docker daemon 镜像加速器（参见下方「常见问题」）

### 5. 验证服务

```bash
# 健康检查
curl http://localhost:8000/api/v1/health

# 发送问答请求
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "如何注册账号？"}'

# 触发知识库重建
curl -X POST http://localhost:8000/api/v1/knowledge/rebuild

# 查看知识库状态
curl http://localhost:8000/api/v1/knowledge/status
```

## 架构

```
用户请求 → FastAPI → 知识检索(Milvus Top20) → 精排(CrossEncoder Top5) → LLM 生成 → 返回
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/chat` | POST | 问答接口 |
| `/api/v1/chat/stream` | POST | SSE 流式问答 |
| `/api/v1/knowledge/rebuild` | POST | 手动触发知识库重建 |
| `/api/v1/knowledge/status` | GET | 知识库状态查询 |
| `/api/v1/health` | GET | 健康检查 |

## 配置说明

所有配置项在 `conf/settings.example.yaml` 中有详细注释。关键配置：

- **llm**: 大模型接入（base_url, api_key, model_name）
- **embedding**: 向量化模型（默认 bge-large-zh-v1.5）
- **milvus**: 向量数据库连接
- **reranker**: 精排模型配置
- **watcher**: 文件监听自动更新配置

## 监控

Grafana Dashboard 预置了以下面板：
- 请求 QPS 和延迟（P95）
- LLM Token 消耗统计
- 知识库检索/精排耗时
- 知识库重建次数
- 错误率

## 目录结构

```
├── app/                  # 应用代码
│   ├── api/              # API 路由和 Schema
│   ├── config/           # 配置加载模块（Python 包）
│   ├── knowledge/        # 知识库（索引、检索、监听）
│   ├── llm/              # LLM 客户端
│   ├── retrieval/        # 精排模块
│   ├── service/          # 业务逻辑
│   └── observability/    # 监控指标
├── conf/                 # 配置文件（settings.yaml）
├── models/               # 预下载的模型文件（.gitignore 忽略）
├── knowledge_docs/       # 知识库文档目录
├── logs/                 # 日志目录
├── monitoring/           # Prometheus & Grafana 配置
├── docker-compose.yml    # 容器编排
├── Dockerfile            # 应用镜像
└── requirements.txt      # Python 依赖
```

## 停止服务

```bash
docker compose down
```

清除所有数据（包括向量库）：

```bash
docker compose down -v
```

## 常见问题

### Docker 镜像拉取慢

配置 Docker daemon 镜像加速器，编辑 Docker Desktop → Settings → Docker Engine：

```json
{
  "registry-mirrors": [
    "https://mirror.ccs.tencentyun.com",
    "https://docker.m.daocloud.io"
  ]
}
```

### pip 安装依赖报 hash 校验错误

Dockerfile 已配置清华 pip 源。如仍有问题，可尝试清除 Docker build cache 重新构建：

```bash
docker builder prune
docker compose build --no-cache app
```

### 模型下载失败（hf-mirror.com 不可达）

在宿主机使用 `hf` CLI 预下载模型到 `./models/` 目录，然后在配置中使用本地路径。模型目录会通过 volume 挂载到容器内 `/app/models/`。

### `huggingface-cli` 命令不存在

新版 huggingface-hub 已将 CLI 更名为 `hf`：

```bash
brew install pipx && pipx install huggingface-hub
hf download BAAI/bge-large-zh-v1.5 --local-dir ./models/bge-large-zh-v1.5
```

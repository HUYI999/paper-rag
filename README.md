# 论文检索 RAG 系统

基于 RAG（检索增强生成）的学术论文智能问答系统。将论文预先存入知识库，提问时精准检索相关片段，避免将整篇论文喂给 LLM 产生幻觉。

## 功能特性

- **混合检索**：向量语义检索 + BM25 关键词检索，通过 RRF 融合，兼顾语义理解与精确匹配
- **多模态支持**：自动提取 PDF 中的图片，用 Vision LLM 生成文字描述，支持"问文找图"
- **可观测 Dashboard**：多页面 Streamlit 界面，支持文件摄入、对话查询、摄入/查询链路追踪、数据浏览
- **MCP 协议接入**：支持 Claude Desktop 等 AI 工具直接调用本地知识库
- **一键部署**：Docker 容器化，交互式 `setup.py` 向导配置后即可运行

## 快速开始

### 1. 克隆项目

```bash
git clone git@github.com:HUYI999/paper-rag.git
cd paper-rag
```

### 2. 交互式配置并启动

```bash
python setup.py
```

按提示依次选择 LLM 提供商、Embedding、Vision、Rerank 和启动方式，脚本自动写入 `.env` 和 `config/settings.yaml`，然后启动服务。

### 3. 访问 Dashboard

浏览器打开 [http://localhost:8501](http://localhost:8501)

---

## Dashboard 页面

| 页面 | 功能 |
|------|------|
| Overview | 查看当前配置（LLM / Embedding / VectorStore）和 Trace 统计 |
| Query | 对话查询，展示答案和引文来源 |
| Data Browser | 浏览已入库的文档和 Chunk 内容 |
| Ingestion Manager | 上传 PDF，带进度条的 6 阶段摄入管道，支持删除文档 |
| Ingestion Traces | 查看每次摄入的详细链路（加载→分块→变换→编码→存储） |
| Query Traces | 查看每次查询的检索路径（Dense / Sparse / RRF / Rerank） |

---

## 系统架构

```
PDF 论文
  ↓
[摄入管道]
  ├─ 解析 PDF → 提取文本 + 图片
  ├─ 文本分块（递归字符分割，chunk=1000，overlap=200）
  ├─ Vision LLM 为图片生成描述，注入对应文本块
  └─ 双路编码存储
       ├─ 向量数据库（ChromaDB）← 语义向量
       └─ BM25 倒排索引（JSON）← 关键词索引

用户提问
  ↓
[查询管道]
  ├─ 向量检索（Dense）← 语义相似
  ├─ BM25 检索（Sparse）← 关键词匹配
  ├─ RRF 融合（score = 1/(k+rank_dense) + 1/(k+rank_sparse)）
  ├─ Rerank 精排（可选，Cross-Encoder 或 LLM）
  └─ LLM 生成回答（基于检索片段）
```

---

## 不用 Docker 直接运行

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 API Key
streamlit run src/observability/dashboard/app.py
```

---

## MCP 协议接入

在 Claude Desktop 配置文件中添加：

```json
{
  "mcpServers": {
    "paper-rag": {
      "command": "python",
      "args": ["src/mcp_server/server.py"],
      "cwd": "/path/to/paper-rag"
    }
  }
}
```

重启 Claude Desktop 后，可在对话中直接调用本地知识库。

---

## 配置说明

编辑 `config/settings.yaml` 调整系统行为：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `llm.provider` | LLM 提供商（openai / ollama / deepseek / azure） | openai |
| `llm.model` | 模型名称 | gpt-4o |
| `embedding.model` | Embedding 模型 | text-embedding-ada-002 |
| `vision_llm.enabled` | 是否解析论文图片 | true |
| `rerank.enabled` | 是否启用精排 | false |
| `rerank.provider` | 精排方式（cross_encoder / llm） | cross_encoder |
| `ingestion.chunk_size` | 文本块大小（字符数） | 1000 |
| `ingestion.chunk_overlap` | 相邻块重叠字符数 | 200 |

---

## 技术栈

Python · ChromaDB · BM25 · OpenAI API · Streamlit · MCP Protocol · Docker

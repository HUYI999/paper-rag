# 论文检索 RAG 系统 - 交互式配置启动脚本
# 用法: python setup.py

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── 工具函数 ──────────────────────────────────────────────────

def ask(prompt, default=None):
    """带默认值的输入提示"""
    hint = f"[默认: {default}] " if default else ""
    val = input(f"{prompt} {hint}> ").strip()
    return val if val else default

def choose(prompt, options):
    """数字选择菜单，返回选项字符串"""
    print(f"\n{prompt}")
    for i, (label, _) in enumerate(options, 1):
        print(f"  [{i}] {label}")
    while True:
        raw = input("> ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][1]
        print(f"  请输入 1~{len(options)} 之间的数字")

def yes_no(prompt, default="n"):
    """y/n 提示"""
    hint = "[Y/n]" if default == "y" else "[y/N]"
    val = input(f"{prompt} {hint} > ").strip().lower()
    if not val:
        return default == "y"
    return val.startswith("y")

def section(title):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


# ── Step 1: LLM 提供商 ────────────────────────────────────────

section("Step 1 / 5  LLM 提供商（用于生成回答）")

llm_provider = choose("选择 LLM 提供商：", [
    ("OpenAI（GPT-4o 等）",  "openai"),
    ("Ollama（本地模型，无需 API Key）", "ollama"),
    ("DeepSeek", "deepseek"),
    ("Azure OpenAI", "azure"),
])

# 默认模型名
LLM_DEFAULTS = {
    "openai":   "gpt-4o",
    "ollama":   "llama3",
    "deepseek": "deepseek-chat",
    "azure":    "gpt-4o",
}

llm_model = ask(f"LLM 模型名", LLM_DEFAULTS[llm_provider])

# 收集认证信息
llm_api_key = ""
llm_base_url = ""
azure_endpoint = ""
azure_api_version = ""
azure_deployment = ""

if llm_provider == "openai":
    llm_api_key = ask("OpenAI API Key（sk-...）", os.getenv("OPENAI_API_KEY", ""))
elif llm_provider == "ollama":
    llm_base_url = ask("Ollama 地址", "http://localhost:11434")
elif llm_provider == "deepseek":
    llm_api_key = ask("DeepSeek API Key")
elif llm_provider == "azure":
    llm_api_key     = ask("Azure OpenAI API Key")
    azure_endpoint  = ask("Azure Endpoint（https://xxx.openai.azure.com/）")
    azure_api_version = ask("API Version", "2024-02-01")
    azure_deployment  = ask("Deployment Name", llm_model)


# ── Step 2: Embedding ─────────────────────────────────────────

section("Step 2 / 5  Embedding（用于向量检索）")

reuse_llm_embedding = yes_no(f"复用 LLM 的 {llm_provider} 作为 Embedding 提供商？", default="y")

if reuse_llm_embedding:
    emb_provider = llm_provider
    emb_api_key  = llm_api_key
    emb_base_url = llm_base_url
    EMB_MODEL_DEFAULTS = {
        "openai":   "text-embedding-ada-002",
        "ollama":   "nomic-embed-text",
        "deepseek": "text-embedding-ada-002",
        "azure":    "text-embedding-ada-002",
    }
    emb_model = ask("Embedding 模型名", EMB_MODEL_DEFAULTS[emb_provider])
else:
    emb_provider = choose("Embedding 提供商：", [
        ("OpenAI", "openai"),
        ("Ollama", "ollama"),
        ("Azure OpenAI", "azure"),
    ])
    EMB_MODEL_DEFAULTS = {"openai": "text-embedding-ada-002", "ollama": "nomic-embed-text", "azure": "text-embedding-ada-002"}
    emb_model   = ask("Embedding 模型名", EMB_MODEL_DEFAULTS[emb_provider])
    emb_api_key = ask("Embedding API Key（留空则复用 LLM key）", llm_api_key) or llm_api_key
    emb_base_url = llm_base_url

# Embedding 维度
EMB_DIM = {"text-embedding-ada-002": 1536, "text-embedding-3-small": 1536, "nomic-embed-text": 768}
emb_dim = EMB_DIM.get(emb_model, 1536)


# ── Step 3: Vision LLM ───────────────────────────────────────

section("Step 3 / 5  Vision LLM（解析 PDF 内的图片）")
print("  启用后，摄入论文时会用 Vision LLM 为图片生成文字描述，支持【搜文找图】")
print("  禁用可加速摄入，但图片内容无法被检索到")

vision_enabled = yes_no("启用 Vision LLM？", default="y")
vision_api_key = llm_api_key  # 默认复用

if vision_enabled and llm_provider not in ("openai", "azure"):
    use_separate_vision = yes_no("当前 LLM 不支持 Vision，需单独配置 OpenAI Vision API Key？", default="y")
    if use_separate_vision:
        vision_api_key = ask("OpenAI API Key（用于 Vision）")


# ── Step 4: Rerank ───────────────────────────────────────────

section("Step 4 / 5  Rerank 精排（提升检索准确率）")
print("  Cross-Encoder：本地运行，无需 API，首次需下载模型（约 70MB）")
print("  LLM：调用已配置的 LLM 打分，更慢但无需额外依赖")

rerank_enabled = yes_no("启用 Rerank？", default="n")
rerank_provider = "none"
rerank_model = ""

if rerank_enabled:
    rerank_provider = choose("Rerank 方式：", [
        ("Cross-Encoder（本地，推荐）", "cross_encoder"),
        ("LLM", "llm"),
    ])
    if rerank_provider == "cross_encoder":
        rerank_model = ask("模型名", "cross-encoder/ms-marco-MiniLM-L-6-v2")


# ── Step 5: 启动方式 ─────────────────────────────────────────

section("Step 5 / 5  启动方式")

launch_mode = choose("选择启动方式：", [
    ("Docker（推荐，环境隔离，需已安装 Docker）", "docker"),
    ("本地运行（需已安装 Python 依赖）", "local"),
])


# ── 写入配置 ──────────────────────────────────────────────────

section("配置摘要")
print(f"  LLM:       {llm_provider} / {llm_model}")
print(f"  Embedding: {emb_provider} / {emb_model} (dim={emb_dim})")
print(f"  Vision:    {'启用' if vision_enabled else '禁用'}")
print(f"  Rerank:    {'启用 - ' + rerank_provider if rerank_enabled else '禁用'}")
print(f"  启动方式:  {launch_mode}")

confirm = yes_no("\n确认写入配置并启动？", default="y")
if not confirm:
    print("已取消。")
    sys.exit(0)

# 写 .env
env_lines = []
if llm_provider == "openai" and llm_api_key:
    env_lines.append(f"OPENAI_API_KEY={llm_api_key}")
elif llm_provider == "deepseek" and llm_api_key:
    env_lines.append(f"DEEPSEEK_API_KEY={llm_api_key}")
elif llm_provider == "azure" and llm_api_key:
    env_lines.append(f"AZURE_OPENAI_API_KEY={llm_api_key}")
    env_lines.append(f"AZURE_OPENAI_ENDPOINT={azure_endpoint}")
    env_lines.append(f"AZURE_OPENAI_API_VERSION={azure_api_version}")
elif llm_provider == "ollama":
    env_lines.append(f"OLLAMA_BASE_URL={llm_base_url}")

# Vision 用独立 key 时追加
if vision_enabled and vision_api_key and vision_api_key != llm_api_key:
    env_lines.append(f"VISION_OPENAI_API_KEY={vision_api_key}")

env_path = ROOT / ".env"
env_path.write_text("\n".join(env_lines) + "\n")
print(f"\n  [OK] 已写入 .env")

# 写 settings.yaml（基于模板替换关键字段）
settings_template = f"""# 论文检索 RAG 系统配置（由 setup.py 生成）

llm:
  provider: "{llm_provider}"
  model: "{llm_model}"
  deployment_name: "{azure_deployment}"
  azure_endpoint: "{azure_endpoint}"
  api_version: "{azure_api_version}"
  api_key: ""
  base_url: "{llm_base_url}"
  temperature: 0.0
  max_tokens: 4096

embedding:
  provider: "{emb_provider}"
  model: "{emb_model}"
  dimensions: {emb_dim}
  azure_endpoint: ""
  deployment_name: ""
  api_version: ""
  api_key: ""
  base_url: "{emb_base_url}"

vision_llm:
  enabled: {"true" if vision_enabled else "false"}
  provider: "openai"
  model: "gpt-4o"
  azure_endpoint: ""
  deployment_name: ""
  api_version: ""
  api_key: ""
  max_image_size: 2048

vector_store:
  provider: "chroma"
  persist_directory: "./data/db/chroma"
  collection_name: "papers"

retrieval:
  dense_top_k: 20
  sparse_top_k: 20
  fusion_top_k: 10
  rrf_k: 60

rerank:
  enabled: {"true" if rerank_enabled else "false"}
  provider: "{rerank_provider}"
  model: "{rerank_model}"
  top_k: 5

evaluation:
  enabled: false
  provider: "custom"
  metrics:
    - "hit_rate"
    - "mrr"
    - "faithfulness"

observability:
  log_level: "INFO"
  trace_enabled: true
  trace_file: "./logs/traces.jsonl"
  structured_logging: true

ingestion:
  chunk_size: 1000
  chunk_overlap: 200
  splitter: "recursive"
  batch_size: 100
  chunk_refiner:
    use_llm: false
  metadata_enricher:
    use_llm: false
"""

settings_path = ROOT / "config" / "settings.yaml"
settings_path.write_text(settings_template)
print(f"  [OK] 已写入 config/settings.yaml")


# ── 启动 ──────────────────────────────────────────────────────

DASHBOARD = "src/observability/dashboard/app.py"

def launch_local():
    """安装依赖并本地启动 Streamlit"""
    print("  正在安装依赖...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"], cwd=ROOT)
    print("  启动 Streamlit...")
    print("  访问: http://localhost:8501\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", DASHBOARD], cwd=ROOT)

print()
if launch_mode == "docker":
    print("  正在构建并启动 Docker 容器...")
    result = subprocess.run(["docker", "compose", "up", "--build", "-d"], cwd=ROOT)
    if result.returncode == 0:
        print("\n  [OK] 启动成功！")
        print("  访问: http://localhost:8501")
    else:
        print("\n  [FAIL] Docker 启动失败（Docker Desktop 可能未运行）")
        print("  配置已保存，建议改用本地模式启动。")
        if yes_no("  是否切换为本地运行？", default="y"):
            launch_local()
        else:
            print("\n  配置已保存。之后可手动启动：")
            print(f"    docker compose up                    # Docker 模式")
            print(f"    streamlit run {DASHBOARD}  # 本地模式")
else:
    launch_local()

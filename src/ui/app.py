# 论文检索 RAG 系统 - 极简 Web 界面
# 左栏：批量上传 PDF；右栏：对话查询

import sys
import tempfile
from pathlib import Path

import streamlit as st

# 确保项目根目录在 sys.path 中
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.settings import load_settings
from src.core.query_engine.hybrid_search import create_hybrid_search
from src.core.query_engine.dense_retriever import create_dense_retriever
from src.core.query_engine.sparse_retriever import create_sparse_retriever
from src.core.query_engine.reranker import create_core_reranker
from src.core.query_engine.query_processor import QueryProcessor
from src.core.trace import TraceContext, TraceCollector
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.libs.llm.llm_factory import LLMFactory


@st.cache_resource
def _load_components():
    """初始化所有检索组件（只在启动时执行一次）"""
    settings = load_settings(str(_ROOT / "config" / "settings.yaml"))
    collection = settings.vector_store.collection_name

    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    embedding_client = EmbeddingFactory.create(settings)
    bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{collection}")

    dense_retriever = create_dense_retriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )
    sparse_retriever = create_sparse_retriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
    sparse_retriever.default_collection = collection

    hybrid_search = create_hybrid_search(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )
    reranker = create_core_reranker(settings=settings)
    llm = LLMFactory.create(settings)

    return settings, hybrid_search, reranker, llm


def _run_query(query: str, top_k: int = 5):
    """执行混合检索 + Rerank，返回结果列表"""
    _, hybrid_search, reranker, _ = _load_components()
    trace = TraceContext(trace_type="query")

    hybrid_result = hybrid_search.search(
        query=query,
        top_k=top_k,
        filters=None,
        trace=trace,
    )
    results = hybrid_result if not hasattr(hybrid_result, "results") else hybrid_result.results

    if results and reranker.is_enabled:
        rerank_result = reranker.rerank(query=query, results=results, top_k=top_k, trace=trace)
        results = rerank_result.results

    TraceCollector().collect(trace)
    return results


def _generate_answer(query: str, results) -> str:
    """将检索结果拼为 context，调用 LLM 生成回答"""
    from src.libs.llm.base_llm import Message

    _, _, _, llm = _load_components()

    context_parts = []
    for i, r in enumerate(results[:5], 1):
        source = (r.metadata or {}).get("source_path", "未知来源")
        context_parts.append(f"[{i}] 来源：{source}\n{r.text[:500]}")
    context = "\n\n".join(context_parts)

    messages = [
        Message(role="system", content=(
            "你是一个学术论文助手。请根据提供的检索片段，给出清晰、完整的回答。"
            "你可以综合多个片段的信息进行归纳总结，但结论必须能从这些片段中得到支撑。"
            "回答要有条理，使用要点或分段，避免过于笼统。"
        )),
        Message(role="user", content=f"以下是从论文中检索到的相关片段：\n\n{context}\n\n请回答：{query}"),
    ]
    response = llm.chat(messages)
    return response.content


def _ingest_files(uploaded_files):
    """批量摄入上传的 PDF 文件"""
    settings, _, _, _ = _load_components()
    # collection 必须与检索端一致，否则数据存入 "default" 但从 "papers" 查不到
    pipeline = IngestionPipeline(settings=settings, collection=settings.vector_store.collection_name)
    results = []

    for f in uploaded_files:
        # 写入临时文件后摄入
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(f.read())
            tmp_path = Path(tmp.name)

        try:
            result = pipeline.run(tmp_path)
            results.append((f.name, True, f"成功：{result.chunk_count} 个片段"))
        except Exception as e:
            results.append((f.name, False, f"失败：{e}"))
        finally:
            tmp_path.unlink(missing_ok=True)

    return results


def _get_paper_list():
    """从向量数据库查询已摄入的论文名称列表"""
    try:
        settings, _, _, _ = _load_components()
        import chromadb
        client = chromadb.PersistentClient(path=settings.vector_store.persist_directory)
        collection = client.get_or_create_collection(settings.vector_store.collection_name)
        # 只拉取元数据，不拉向量，节省内存
        data = collection.get(include=["metadatas"])
        seen = set()
        names = []
        for meta in (data.get("metadatas") or []):
            path = (meta or {}).get("source_path", "")
            name = Path(path).name if path else ""
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        return names
    except Exception:
        return []


# ── 页面布局 ──────────────────────────────────────────────────

st.set_page_config(page_title="论文 RAG 检索", layout="wide")
st.title("论文检索 RAG 系统")
st.caption("上传论文 PDF，用自然语言提问，获取精准答案")

left, right = st.columns([1, 2])

# 用 key 计数器实现上传后自动清空文件选择框
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

# ── 左栏：批量上传 ───────────────────────────────────────────

with left:
    st.subheader("上传论文")
    uploaded = st.file_uploader(
        "支持批量上传 PDF",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.upload_key}",
    )

    if uploaded and st.button("开始导入", type="primary", use_container_width=True):
        with st.spinner("正在处理..."):
            ingest_results = _ingest_files(uploaded)

        for name, ok, msg in ingest_results:
            if ok:
                st.success(f"{name}：{msg}")
            else:
                st.error(f"{name}：{msg}")

        # 导入完成后清空上传框，并刷新论文列表
        st.session_state.upload_key += 1
        st.rerun()

    # 显示数据库中已有的论文列表
    st.divider()
    st.caption("已导入的论文")
    papers = _get_paper_list()
    if papers:
        for p in papers:
            st.text(f"• {p}")
    else:
        st.caption("暂无论文，请先上传")

# ── 右栏：对话查询 ───────────────────────────────────────────

with right:
    st.subheader("提问")

    # 初始化对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 用户输入
    if prompt := st.chat_input("输入你的问题，例如：这篇论文的核心方法是什么？"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("检索中..."):
                try:
                    results = _run_query(prompt, top_k=5)

                    if not results:
                        answer = "未找到相关内容，请先上传论文。"
                        st.markdown(answer)
                    else:
                        answer = _generate_answer(prompt, results)
                        st.markdown(answer)

                        # 显示引文来源
                        with st.expander("引文来源"):
                            for i, r in enumerate(results[:5], 1):
                                source = (r.metadata or {}).get("source_path", "未知")
                                page = (r.metadata or {}).get("page_num", "")
                                page_str = f"第 {page} 页" if page else ""
                                st.markdown(f"**[{i}]** {Path(source).name} {page_str}")
                                st.caption(r.text[:200] + "...")

                except Exception as e:
                    answer = f"查询出错：{e}"
                    st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

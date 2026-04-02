# 查询面板 - 混合检索 + LLM 生成回答

from __future__ import annotations

from pathlib import Path

import streamlit as st


@st.cache_resource
def _load_query_components():
    """初始化检索组件（只在首次调用时执行）"""
    from src.core.settings import load_settings
    from src.core.query_engine.hybrid_search import create_hybrid_search
    from src.core.query_engine.dense_retriever import create_dense_retriever
    from src.core.query_engine.sparse_retriever import create_sparse_retriever
    from src.core.query_engine.reranker import create_core_reranker
    from src.core.query_engine.query_processor import QueryProcessor
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory
    from src.libs.llm.llm_factory import LLMFactory

    settings = load_settings()
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


def _run_query(query: str, top_k: int):
    from src.core.trace import TraceContext, TraceCollector

    _, hybrid_search, reranker, _ = _load_query_components()
    trace = TraceContext(trace_type="query")

    hybrid_result = hybrid_search.search(query=query, top_k=top_k, filters=None, trace=trace)
    results = hybrid_result if not hasattr(hybrid_result, "results") else hybrid_result.results

    if results and reranker.is_enabled:
        rerank_result = reranker.rerank(query=query, results=results, top_k=top_k, trace=trace)
        results = rerank_result.results

    TraceCollector().collect(trace)
    return results


def _generate_answer(query: str, results) -> str:
    from src.libs.llm.base_llm import Message

    _, _, _, llm = _load_query_components()

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


def render() -> None:
    st.header("💬 Query")

    top_k = st.sidebar.slider("返回结果数 (top_k)", min_value=1, max_value=20, value=5)

    if "query_messages" not in st.session_state:
        st.session_state.query_messages = []

    # 显示历史消息
    for msg in st.session_state.query_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("输入问题，例如：这篇论文的核心方法是什么？"):
        st.session_state.query_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("检索中..."):
                try:
                    results = _run_query(prompt, top_k=top_k)

                    if not results:
                        answer = "未找到相关内容，请先在 Ingestion Manager 上传论文。"
                        st.markdown(answer)
                    else:
                        answer = _generate_answer(prompt, results)
                        st.markdown(answer)

                        with st.expander(f"引文来源（共 {len(results)} 条）"):
                            for i, r in enumerate(results, 1):
                                source = (r.metadata or {}).get("source_path", "未知")
                                page = (r.metadata or {}).get("page_num", "")
                                page_str = f"第 {page} 页" if page else ""
                                st.markdown(f"**[{i}]** `{Path(source).name}` {page_str}")
                                st.caption(r.text[:200] + "...")

                except Exception as e:
                    answer = f"查询出错：{e}"
                    st.error(answer)

        st.session_state.query_messages.append({"role": "assistant", "content": answer})

    if st.session_state.query_messages:
        if st.button("清空对话", use_container_width=True):
            st.session_state.query_messages = []
            st.rerun()

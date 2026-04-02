"""Modular RAG Dashboard – multi-page Streamlit application.

Entry-point: ``streamlit run src/observability/dashboard/app.py``

Pages are registered via ``st.navigation()`` and rendered by their
respective modules under ``pages/``.  Pages not yet implemented show
a placeholder message.
"""

from __future__ import annotations

import os
from pathlib import Path

# 启动时加载 .env，让 OPENAI_API_KEY 等环境变量生效
_env_file = Path(__file__).resolve().parents[3] / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import streamlit as st


# ── Page definitions ─────────────────────────────────────────────────

def _page_overview() -> None:
    from src.observability.dashboard.pages.overview import render
    render()


def _page_data_browser() -> None:
    from src.observability.dashboard.pages.data_browser import render
    render()


def _page_ingestion_manager() -> None:
    from src.observability.dashboard.pages.ingestion_manager import render
    render()


def _page_ingestion_traces() -> None:
    from src.observability.dashboard.pages.ingestion_traces import render
    render()


def _page_query_traces() -> None:
    from src.observability.dashboard.pages.query_traces import render
    render()


def _page_query_panel() -> None:
    from src.observability.dashboard.pages.query_panel import render
    render()


# ── Navigation ───────────────────────────────────────────────────────

pages = [
    st.Page(_page_overview, title="Overview", icon="📊", default=True),
    st.Page(_page_query_panel, title="Query", icon="💬"),
    st.Page(_page_data_browser, title="Data Browser", icon="🔍"),
    st.Page(_page_ingestion_manager, title="Ingestion Manager", icon="📥"),
    st.Page(_page_ingestion_traces, title="Ingestion Traces", icon="🔬"),
    st.Page(_page_query_traces, title="Query Traces", icon="🔎"),
]


def main() -> None:
    st.set_page_config(
        page_title="Modular RAG Dashboard",
        page_icon="📊",
        layout="wide",
    )

    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
else:
    # When run directly via `streamlit run app.py`
    main()

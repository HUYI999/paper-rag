"""Overview page – system configuration and trace statistics."""

from __future__ import annotations

import streamlit as st

from src.observability.dashboard.services.config_service import ConfigService


def render() -> None:
    """Render the Overview page."""
    st.header("📊 System Overview")

    # ── Component configuration cards ──────────────────────────────
    st.subheader("🔧 Component Configuration")

    try:
        config_service = ConfigService()
        cards = config_service.get_component_cards()
    except Exception as exc:
        st.error(f"Failed to load configuration: {exc}")
        return

    cols = st.columns(min(len(cards), 3))
    for idx, card in enumerate(cards):
        with cols[idx % len(cols)]:
            st.markdown(f"**{card.name}**")
            st.caption(f"Provider: `{card.provider}`  \nModel: `{card.model}`")
            with st.expander("Details"):
                for k, v in card.extra.items():
                    st.text(f"{k}: {v}")

    # ── Trace file statistics ──────────────────────────────────────
    st.subheader("📈 Trace Statistics")

    from src.core.settings import resolve_path
    traces_path = resolve_path("logs/traces.jsonl")
    if traces_path.exists():
        line_count = sum(1 for _ in traces_path.open(encoding="utf-8"))
        if line_count > 0:
            st.metric("Total traces", line_count)
        else:
            st.info("No traces recorded yet. Run a query or ingestion first.")
    else:
        st.info("No traces recorded yet. Run a query or ingestion first.")

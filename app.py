"""
Darukaa.Earth — AI Biodiversity Intelligence

Streamlit interface. Deliberately plain: the challenge asks for depth of
reasoning, not UI. The interface exists to make the reasoning inspectable —
every panel shows what the engine did, not just what it concluded.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json

import streamlit as st

from core.conversation import Session
from core.render import to_json_string, to_markdown
from core.retrieve import kb_is_ready, kb_stats, search
from core.schema import CORE_ENV_METRICS

st.set_page_config(page_title="Darukaa.Earth Biodiversity Intelligence", layout="wide")

EXAMPLES = {
    "Semi-arid monoculture wheat (challenge example)":
        "Soil organic carbon 0.3%, rainfall low, monoculture wheat, semi-arid region, mean temperature 28 C.",
    "Acid soil with declining biodiversity":
        "pH 5.1, SOC 0.8%, monoculture maize, biodiversity declining, no hedgerows, rainfall 900mm.",
    "Heavy chemical use, simplified landscape":
        "We spray insecticide weekly, biodiversity is declining, no field margins, monoculture cotton, SOC 1.1%.",
    "Recently cleared land":
        "We cleared about 40% of the tree cover three years ago. Species richness is low, habitat diversity is low, land is now grazing land, rainfall 700mm.",
    "Structured JSON input":
        '{"soil_ph": 6.4, "soc": 0.9, "rainfall": 380, "land_use": "monoculture wheat", "pollution": "high", "species_richness": "low", "temperature": 29}',
}


def get_session() -> Session:
    if "session" not in st.session_state:
        st.session_state.session = Session()
    return st.session_state.session


def main() -> None:
    st.title("Darukaa.Earth — AI Biodiversity Intelligence")
    st.caption(
        "An environmental reasoning engine with a retrieval-grounded evidence layer. "
        "Recommendations are produced by an explicit multi-variable rule graph and are "
        "supported only by passages retrieved from the indexed scientific knowledge base."
    )

    if not kb_is_ready():
        st.error("Knowledge base not built. Run `python -m knowledge.ingest` and reload.")
        st.stop()

    session = get_session()

    # ------------------------------------------------------------------ sidebar
    with st.sidebar:
        st.header("Knowledge base")
        stats = kb_stats()
        c1, c2 = st.columns(2)
        c1.metric("Sources", stats["sources"])
        c2.metric("Chunks", stats["chunks"])
        st.caption(f"Hybrid retrieval: dense LSA vectors (dim {stats['dim']}) + BM25, fused with RRF.")

        st.divider()
        st.header("Environmental state")
        known = session.state.known_core_metrics()
        st.progress(min(len(known) / 5, 1.0), text=f"{len(known)} of {len(CORE_ENV_METRICS)} variables")
        if session.state.observations:
            for line in session.state.summary_lines():
                st.write(f"- {line}")
        else:
            st.caption("Nothing recorded yet.")

        st.divider()
        if st.button("Reset session", use_container_width=True):
            session.reset()
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.header("Search the knowledge base")
        probe = st.text_input("Query", placeholder="e.g. agroforestry arid soil carbon")
        if probe:
            for item in search(probe, k=3):
                with st.expander(f"{item.organisation} — {item.section}"):
                    st.write(item.text)
                    st.caption(f"{item.title} · retrieval: {item.retrieval} · score {item.score}")

    # ------------------------------------------------------------------ examples
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.subheader("Try an example")
        cols = st.columns(len(EXAMPLES))
        for col, (label, text) in zip(cols, EXAMPLES.items()):
            if col.button(label, use_container_width=True):
                st.session_state.pending_input = text
                st.rerun()

    # ------------------------------------------------------------------ history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("json"):
                with st.expander("Structured output (JSON)"):
                    st.code(msg["json"], language="json")

    # ------------------------------------------------------------------ input
    user_input = st.chat_input("Describe your land, or paste structured JSON…")
    if "pending_input" in st.session_state:
        user_input = st.session_state.pop("pending_input")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Reasoning across variables and retrieving evidence…"):
                response = session.process(user_input)
                markdown = to_markdown(response)
                payload = to_json_string(response)
            st.markdown(markdown)
            with st.expander("Structured output (JSON)"):
                st.code(payload, language="json")
            st.download_button(
                "Download this analysis (JSON)",
                data=payload,
                file_name="darukaa_analysis.json",
                mime="application/json",
            )

        st.session_state.messages.append(
            {"role": "assistant", "content": markdown, "json": payload}
        )
        st.rerun()


if __name__ == "__main__":
    main()

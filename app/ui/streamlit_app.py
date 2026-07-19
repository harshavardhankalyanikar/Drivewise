"""
DriveWise Streamlit UI.

Run with:
    streamlit run app/ui/streamlit_app.py

Features: brand/model/variant/section metadata filters, a chat-style Q&A
interface, an expandable "Retrieved Chunks" panel (with hybrid + re-rank
scores) for transparency, a Sources panel with document/page citations, a
confidence indicator, and a brochure upload widget that triggers re-indexing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.chains.pipeline import get_rag_chain, rebuild_index_from_brochures  # noqa: E402
from app.config.schemas import QueryRequest  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.utils.persistence import load_chunks  # noqa: E402

st.set_page_config(page_title="DriveWise", page_icon="🚗", layout="wide")

CUSTOM_CSS = """
<style>
.stApp { background-color: #f7f9fc; }
.drivewise-header { font-size: 2rem; font-weight: 700; color: #0B3D91; margin-bottom: 0; }
.drivewise-subheader { color: #555; margin-top: 0; margin-bottom: 1.2rem; }
.source-card {
    background: white; border-left: 4px solid #0B3D91; border-radius: 6px;
    padding: 0.6rem 0.9rem; margin-bottom: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.confidence-badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _get_metadata_facets():
    try:
        chunks = load_chunks()
    except FileNotFoundError:
        return None
    brand_to_models: dict[str, set] = {}
    sections = set()
    for c in chunks:
        brand_to_models.setdefault(c.metadata.car_brand, set()).add(c.metadata.car_model)
        sections.add(c.metadata.section)
    return {
        "brand_to_models": {k: sorted(v) for k, v in brand_to_models.items()},
        "sections": sorted(sections),
    }


def _confidence_badge(confidence: float) -> str:
    if confidence >= 0.7:
        color, label = "#1B5E20", "High"
    elif confidence >= 0.4:
        color, label = "#E65100", "Medium"
    else:
        color, label = "#B71C1C", "Low"
    return (
        f'<span class="confidence-badge" style="background:{color}20;color:{color};">'
        f"{label} confidence ({confidence:.2f})</span>"
    )


st.markdown('<p class="drivewise-header">🚗 DriveWise</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="drivewise-subheader">Metadata-aware, brochure-grounded automotive assistant. '
    "Ask about mileage, engine specs, safety features, variants, and more.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("🔎 Filters")
    facets = _get_metadata_facets()

    if facets is None:
        st.warning("No brochure index found yet. Upload a PDF below or run `python scripts/ingest.py`.")
        selected_brand = None
        selected_model = None
        selected_section = None
    else:
        brand_options = ["Any"] + sorted(facets["brand_to_models"].keys())
        selected_brand = st.selectbox("Car Brand", brand_options)
        if selected_brand != "Any":
            model_options = ["Any"] + facets["brand_to_models"][selected_brand]
        else:
            model_options = ["Any"]
        selected_model = st.selectbox("Car Model", model_options)
        section_options = ["Any"] + facets["sections"]
        selected_section = st.selectbox("Section", section_options)

    st.divider()
    st.header("📄 Upload Brochure")
    uploaded_file = st.file_uploader("Add a new car brochure PDF", type=["pdf"])
    if uploaded_file is not None and st.button("Upload & Re-index"):
        dest = Path(settings.brochures_dir) / uploaded_file.name
        with open(dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Re-indexing brochures (parsing, chunking, embedding)..."):
            try:
                count = rebuild_index_from_brochures()
                st.cache_data.clear()
                st.success(f"Indexed {count} chunks. Ready to query!")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Re-indexing failed: {exc}")

    st.divider()
    st.header("⚙️ Retrieval Settings")
    top_k = st.slider("Chunks to retrieve before re-ranking", min_value=5, max_value=30, value=settings.retrieval_top_k)

tab_chat, tab_eval = st.tabs(["💬 Chatbot Q&A", "📊 Evaluation Dashboard"])

with tab_chat:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            st.markdown(_confidence_badge(turn["confidence"]), unsafe_allow_html=True)
            if turn["sources"]:
                with st.expander(f"📚 Sources ({len(turn['sources'])})"):
                    for s in turn["sources"]:
                        st.markdown(
                            f'<div class="source-card"><b>{s["car_brand"]} {s["car_model"]}</b> — '
                            f'{s["document_name"]} · Section: {s["section"]} · Page {s["page"]}'
                            f'<br><i>{s["snippet"]}</i></div>',
                            unsafe_allow_html=True,
                        )
            with st.expander("🧠 Reasoning summary"):
                st.write(turn["reasoning_summary"])

    question = st.chat_input("Ask about mileage, safety, engine specs, sunroof, ADAS...")

    if question:
        st.session_state.chat_history.append({"question": question, "answer": None})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving brochure context and generating a grounded answer..."):
                try:
                    chain = get_rag_chain()
                    request = QueryRequest(
                        question=question,
                        car_brand=None if selected_brand in (None, "Any") else selected_brand,
                        car_model=None if selected_model in (None, "Any") else selected_model,
                        section=None if selected_section in (None, "Any") else selected_section,
                        top_k=top_k,
                    )
                    response = chain.answer(request)

                    st.write(response.answer)
                    st.markdown(_confidence_badge(response.confidence), unsafe_allow_html=True)

                    if response.sources:
                        with st.expander(f"📚 Sources ({len(response.sources)})", expanded=False):
                            for s in response.sources:
                                st.markdown(
                                    f'<div class="source-card"><b>{s.car_brand} {s.car_model}</b> — '
                                    f"{s.document_name} · Section: {s.section} · Page {s.page}"
                                    f"<br><i>{s.snippet}</i></div>",
                                    unsafe_allow_html=True,
                                )
                    with st.expander("🧠 Reasoning summary"):
                        st.write(response.reasoning_summary)

                    st.session_state.chat_history[-1] = {
                        "question": question,
                        "answer": response.answer,
                        "confidence": response.confidence,
                        "sources": [s.model_dump() for s in response.sources],
                        "reasoning_summary": response.reasoning_summary,
                    }
                except FileNotFoundError:
                    st.error("No index found yet. Upload a brochure PDF in the sidebar, or run `python scripts/ingest.py`.")
                    st.session_state.chat_history.pop()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Something went wrong: {exc}")
                    st.session_state.chat_history.pop()

with tab_eval:
    st.header("📊 RAG Evaluation Dashboard")

    col_run, col_clear = st.columns([1, 4])

    eval_csv = Path(settings.evaluation_results_path)
    if not eval_csv.exists():
        st.info("No evaluation results file found yet. Click 'Run Evaluation Suite' to generate results.")
    else:
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go

        df = pd.read_csv(eval_csv)
        if df.empty:
            st.warning("Evaluation results file is empty.")
        else:
            st.subheader("⏱️ Latency Breakdown per Query")
            if "Retrieval Time" in df.columns and "Reranking Time" in df.columns and "LLM Generation Time" in df.columns:
                fig_latency = go.Figure()
                fig_latency.add_trace(go.Bar(
                    x=df.index + 1,
                    y=df["Retrieval Time"],
                    name="Retrieval Time",
                    marker_color="#1f77b4"
                ))
                fig_latency.add_trace(go.Bar(
                    x=df.index + 1,
                    y=df["Reranking Time"],
                    name="Re-ranking Time",
                    marker_color="#ff7f0e"
                ))
                fig_latency.add_trace(go.Bar(
                    x=df.index + 1,
                    y=df["LLM Generation Time"],
                    name="LLM Generation Time",
                    marker_color="#2ca02c"
                ))
                fig_latency.update_layout(
                    barmode="stack",
                    xaxis_title="Query index",
                    yaxis_title="Latency (ms)",
                    legend_title="Pipeline Stage",
                    hovermode="x"
                )
                st.plotly_chart(fig_latency, use_container_width=True)
            else:
                fig_latency = px.bar(df, x=df.index + 1, y="Latency", title="Total Latency per Query")
                fig_latency.update_layout(xaxis_title="Query index", yaxis_title="Latency (ms)")
                st.plotly_chart(fig_latency, use_container_width=True)

            


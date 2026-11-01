"""Streamlit demo for the FastAPI-backed Related Work Research Assistant."""

import html
import json
import os
from typing import Any

import httpx
import streamlit as st

API_URL = os.getenv("RWA_API_URL", "http://127.0.0.1:8000").rstrip("/")
PAIR_COLORS = ["#FFD54F", "#64B5F6", "#81C784"]

st.set_page_config(
    page_title="Related Work Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px;}
    [data-testid="stSidebar"] {background: #f7f8fb;}
    .hero {padding: 1.2rem 1.4rem; border: 1px solid #e3e7ef; border-radius: 16px;
           background: linear-gradient(135deg, #f8fbff, #f6f3ff); margin-bottom: 1.2rem;}
    .hero h1 {margin: 0; color: #182238; font-size: 2rem;}
    .hero p {margin: .4rem 0 0; color: #526078;}
    .evidence-card {border: 1px solid #e2e7f0; border-left: 6px solid var(--pair-color);
                    border-radius: 12px; padding: .9rem 1rem; margin: .6rem 0; background: white;}
    .score {font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85rem;
            color: #46546b;}
    .legend-dot {display:inline-block; width:14px; height:14px; border-radius:4px;
                 margin-right:7px; vertical-align:-2px; border:1px solid rgba(0,0,0,.12);}
    </style>
    """,
    unsafe_allow_html=True,
)


class APIError(RuntimeError):
    pass


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
        return str(payload.get("detail") or payload)
    except Exception:
        return response.text or f"HTTP {response.status_code}"


def api_post(path: str, payload: dict[str, Any]) -> Any:
    try:
        response = httpx.post(f"{API_URL}{path}", json=payload, timeout=30)
    except httpx.HTTPError as exc:
        raise APIError(f"FastAPI backend is unavailable at {API_URL}: {exc}") from exc
    if response.is_error:
        raise APIError(_error_message(response))
    api_get.clear()
    return response.json()


def api_get_live(path: str) -> Any:
    """Read volatile job state without the five-minute retrieval cache."""
    try:
        response = httpx.get(f"{API_URL}{path}", timeout=30)
    except httpx.HTTPError as exc:
        raise APIError(f"FastAPI backend is unavailable at {API_URL}: {exc}") from exc
    if response.is_error:
        raise APIError(_error_message(response))
    return response.json()


def api_get_optional_live(path: str) -> Any | None:
    try:
        response = httpx.get(f"{API_URL}{path}", timeout=30)
    except httpx.HTTPError as exc:
        raise APIError(f"FastAPI backend is unavailable at {API_URL}: {exc}") from exc
    if response.status_code == 404:
        return None
    if response.is_error:
        raise APIError(_error_message(response))
    return response.json()


def related_work_panel(paper_id: str) -> None:
    st.subheader("Related Work")
    notice_key = f"generation_notice_{paper_id}"
    if notice := st.session_state.pop(notice_key, None):
        st.success(notice)
    try:
        status_payload = api_get_live(f"/papers/{paper_id}/related-work")
    except APIError as exc:
        st.error(str(exc))
        return

    labels = {
        "PRIMERA — summary baseline": "primera",
        "Qwen 2.5 0.5B": "qwen",
        "Qwen + LoRA": "qwen_lora",
    }
    selected_label = st.radio(
        "Generator",
        list(labels),
        horizontal=True,
        key=f"related_work_model_{paper_id}",
    )
    selected_key = labels[selected_label]
    model = next(item for item in status_payload["models"] if item["model_key"] == selected_key)
    if not model["available"]:
        st.info(
            "Qwen + LoRA is not available yet. Complete LoRA training first. "
            "No inference was started."
        )
        if model.get("unavailable_reason"):
            st.caption(model["unavailable_reason"])
        return
    st.caption(f"Database status: `{model['status']}`")
    if selected_key == "primera":
        st.info(
            "PRIMERA is a Multi-XScience summarization baseline, not an instruction-tuned "
            "generator. It may emit the literal word 'cite' and cannot reliably preserve "
            "source-specific citation keys."
        )
    if model["status"] in {"pending", "running"}:
        active_job = model.get("active_job") or {}
        job_id = active_job.get("job_id", "unknown")
        if model["status"] == "pending":
            st.info(
                f"Job `{job_id}` is queued. Start or keep the generation worker running, "
                "then use Refresh generation status."
            )
        else:
            st.info(
                f"Job `{job_id}` is currently running in the background worker. "
                "The result will appear after it is persisted to PostgreSQL."
            )
        if active_job.get("created_at"):
            st.caption(
                f"Queued: {active_job['created_at']}"
                + (
                    f" · started: {active_job['started_at']}"
                    if active_job.get("started_at")
                    else ""
                )
            )
    latest = model.get("latest")
    if latest:
        with st.expander("Model input data"):
            st.caption(
                f"Prompt {latest['prompt_version']} · input hash {latest['input_hash']} · "
                f"{latest['input_token_count']} tokenizer tokens"
            )
            rendered = latest.get("rendered_model_input")
            input_tab, evidence_tab, config_tab = st.tabs(
                ["Rendered model input", "Evidence snapshot", "Generation config"]
            )
            with input_tab:
                if rendered and rendered.get("messages"):
                    for message in rendered["messages"]:
                        st.markdown(f"**{message['role'].title()} message**")
                        st.code(message["content"], language=None)
                elif rendered and rendered.get("source"):
                    st.code(rendered["source"], language=None)
                else:
                    st.info(
                        "The exact rendered prompt is unavailable for this historical prompt "
                        "version. Its persisted evidence is available in the next tab."
                    )
            with evidence_tab:
                st.json(latest.get("evidence_snapshot") or {})
            with config_tab:
                st.json(latest.get("generation_config") or {})
        if latest["status"] == "completed":
            st.markdown(latest["generated_text"])
            st.code(latest["generated_text"], language=None)
            st.caption(
                f"Input {latest['input_token_count']} tokens · output {latest['output_token_count']} · "
                f"model revision {latest['model_revision'][:12]} · prompt {latest['prompt_version']}"
            )
            with st.expander("Citations and evidence references"):
                st.write("Citations:", latest["citations"] or "No bracketed citations produced")
                st.json(latest["source_references"])
        else:
            if model["status"] in {"pending", "running"}:
                st.caption("The diagnostic below belongs to the previous saved generation.")
            st.warning(
                "The generated text was saved but rejected by validation: "
                + str(latest.get("error_message") or latest["status"])
            )
            if latest.get("validation_errors"):
                st.write(latest["validation_errors"])
            if latest.get("generated_text"):
                with st.expander("Show rejected generated text (debug)"):
                    st.write(latest["generated_text"])
                    st.caption(
                        f"Generation {latest['generation_id']} · input "
                        f"{latest['input_token_count']} · output {latest['output_token_count']} · "
                        f"finish {latest.get('finish_reason') or 'unknown'} · "
                        f"prompt {latest['prompt_version']}"
                    )
    if model["status"] == "missing" and st.button(
        "Generate selected model", key=f"generate_{paper_id}_{selected_key}"
    ):
        try:
            api_post(
                f"/papers/{paper_id}/related-work/generate",
                {"models": [selected_key]},
            )
            st.rerun()
        except APIError as exc:
            st.error(str(exc))
    if st.button("Regenerate selected model", key=f"regenerate_{paper_id}_{selected_key}"):
        try:
            response = api_post(
                f"/papers/{paper_id}/related-work/regenerate",
                {"models": [selected_key]},
            )
            job_ids = response.get("job_ids", [])
            st.session_state[notice_key] = (
                f"Regeneration queued. Job ID: {job_ids[0]}"
                if job_ids
                else "A generation job for this model is already pending or running."
            )
            st.rerun()
        except APIError as exc:
            st.error(str(exc))
    if st.button("Refresh generation status", key=f"refresh_generation_{paper_id}"):
        api_get.clear()
        st.rerun()


def _display_metric(value: Any, *, percent: bool = False, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    number = float(value)
    return f"{number * 100:.1f}%" if percent else f"{number:.4f}{suffix}"


def generation_evaluation_panel(paper_id: str) -> None:
    st.subheader("Generation Evaluation")
    st.caption(
        "Read-only view of explicitly executed evaluation runs. Streamlit never starts "
        "benchmark inference or NLI automatically."
    )
    benchmark_tab, evidence_tab = st.tabs(
        ["A · Multi-XScience Benchmark", "B · Evidence Grounding"]
    )
    with benchmark_tab:
        try:
            benchmark = api_get_optional_live("/evaluations/generation/benchmark/latest")
            benchmark_history = api_get_live("/evaluations/generation/runs?kind=a&limit=20")
        except APIError as exc:
            st.error(str(exc))
            benchmark = None
            benchmark_history = []
        if benchmark is None:
            st.info(
                "No Evaluation A run is saved. Run `uv run related-work "
                "evaluate-generation-a --limit 10 --seed 42`."
            )
        else:
            if benchmark_history:
                selected_run_id = st.selectbox(
                    "Saved benchmark run",
                    [run["run_id"] for run in benchmark_history],
                    key="generation_evaluation_a_run",
                )
                benchmark = next(
                    run for run in benchmark_history if run["run_id"] == selected_run_id
                )
            labels = {"primera": "PRIMERA", "qwen": "Qwen", "qwen_lora": "Qwen + LoRA"}
            rows = []
            for metric, key in (("ROUGE-1", "rouge1"), ("ROUGE-2", "rouge2"), ("ROUGE-L", "rougeL")):
                rows.append(
                    {
                        "Metric": metric,
                        **{
                            label: _display_metric(benchmark.get("models", {}).get(model, {}).get(key))
                            for model, label in labels.items()
                        },
                    }
                )
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption(
                f"Run `{benchmark['run_id']}` · status `{benchmark['status']}` · "
                f"{len(benchmark.get('sample_ids', []))} examples · "
                f"seed {benchmark.get('config', {}).get('seed')}"
            )
            with st.expander("Benchmark configuration and model status"):
                st.json({"config": benchmark.get("config"), "models": benchmark.get("models")})
            examples = benchmark.get("examples", [])
            if examples:
                selected = st.selectbox(
                    "Benchmark example",
                    examples,
                    format_func=lambda item: item["sample_id"],
                    key=f"benchmark_example_{benchmark['run_id']}",
                )
                st.markdown("**Human-written Related Work**")
                st.write(selected["ground_truth_original"])
                columns = st.columns(3)
                for column, (model, label) in zip(columns, labels.items(), strict=True):
                    with column:
                        prediction = selected["predictions"].get(model, {})
                        st.markdown(f"**{label}**")
                        st.caption(f"Status: {prediction.get('status', 'not evaluated')}")
                        st.write(prediction.get("text_original") or prediction.get("text") or "N/A")
                        if prediction.get("rouge"):
                            st.json(prediction["rouge"])

    with evidence_tab:
        try:
            evidence = api_get_optional_live(
                f"/evaluations/generation/evidence/{paper_id}/latest"
            )
            fixed = api_get_optional_live("/evaluations/generation/evidence/fixed/latest")
            evidence_history = api_get_live("/evaluations/generation/runs?kind=b&limit=100")
        except APIError as exc:
            st.error(str(exc))
            evidence = fixed = None
            evidence_history = []
        paper_history = [
            {"run": run, "paper": paper}
            for run in evidence_history
            for paper in run.get("papers", [])
            if str(paper.get("paper_id")) == paper_id
        ]
        if evidence is None:
            st.info(
                "This paper has not been evaluated. Run `uv run related-work "
                f"evaluate-generation-b {paper_id}`."
            )
        else:
            if paper_history:
                selected_run_id = st.selectbox(
                    "Saved evidence-grounding run",
                    [item["run"]["run_id"] for item in paper_history],
                    key=f"generation_evaluation_b_run_{paper_id}",
                )
                selected_history = next(
                    item for item in paper_history if item["run"]["run_id"] == selected_run_id
                )
                evidence = {
                    "run": {
                        key: value
                        for key, value in selected_history["run"].items()
                        if key != "papers"
                    },
                    "paper": selected_history["paper"],
                }
            paper = evidence["paper"]
            labels = {"primera": "PRIMERA", "qwen": "Qwen", "qwen_lora": "Qwen + LoRA"}
            metric_rows = []
            definitions = [
                ("Citation validity", lambda model: (model.get("citation_metrics") or {}).get("validity"), True, ""),
                ("Citation coverage", lambda model: (model.get("citation_metrics") or {}).get("coverage"), True, ""),
                ("NLI entailment", lambda model: (model.get("nli") or {}).get("entailment"), True, ""),
                ("NLI neutral", lambda model: (model.get("nli") or {}).get("neutral"), True, ""),
                ("NLI contradiction", lambda model: (model.get("nli") or {}).get("contradiction"), True, ""),
                ("Latency", lambda model: model.get("inference_latency_ms"), False, " ms"),
                ("Peak memory", lambda model: model.get("peak_memory_mb"), False, " MB"),
            ]
            for name, getter, percent, suffix in definitions:
                metric_rows.append(
                    {
                        "Metric": name,
                        **{
                            label: _display_metric(
                                getter(paper["models"][model]), percent=percent, suffix=suffix
                            )
                            for model, label in labels.items()
                        },
                    }
                )
            st.dataframe(metric_rows, hide_index=True, width="stretch")
            identical = paper.get("qwen_vs_qwen_lora_identical_input")
            if identical is True:
                st.success("Qwen and Qwen + LoRA used identical persisted prompts and evidence.")
            elif identical is False:
                st.warning("Qwen and Qwen + LoRA did not use identical persisted inputs.")
            else:
                st.info("Direct Qwen/LoRA input comparison is unavailable.")
            st.caption(
                "PRIMERA is an abstract-only complete-system baseline. NLI is an approximate "
                "grounding proxy, not proof of scientific correctness."
            )
            for model_key in ("qwen", "qwen_lora"):
                model = paper["models"][model_key]
                with st.expander(f"{labels[model_key]} claims and cited-source evidence"):
                    if not model.get("claims"):
                        st.info(f"No evaluated claims: {model.get('status')}")
                    for claim in model.get("claims", []):
                        st.markdown(f"**{claim['text']}**")
                        st.caption(
                            f"Citation [{claim['citation']}] · link status `{claim['status']}`"
                        )
                        if claim.get("nli"):
                            st.json(claim["nli"])
                        for evidence_item in claim.get("evidence_scores", []):
                            st.write(evidence_item["text"])
                            st.caption(
                                f"Chunk {evidence_item['chunk_id']} · NLI {evidence_item['nli']}"
                            )
                        st.divider()
        st.markdown("#### Fixed-set aggregate")
        if fixed is None:
            st.info(
                "No fixed-set run is saved. Run `uv run related-work "
                "evaluate-generation-b-fixed --limit 10 --seed 42`."
            )
        else:
            aggregate_rows = []
            for model_key, label in {
                "primera": "PRIMERA",
                "qwen": "Qwen",
                "qwen_lora": "Qwen + LoRA",
            }.items():
                aggregate_rows.append({"Model": label, **fixed.get("aggregate", {}).get(model_key, {})})
            st.dataframe(aggregate_rows, hide_index=True, width="stretch")
            st.caption(
                f"Run `{fixed['run_id']}` · {len(fixed.get('sample_ids', []))} fixed target papers"
            )


@st.cache_data(ttl=300, show_spinner=False)
def api_get(path: str, params: tuple[tuple[str, Any], ...] = ()) -> Any:
    try:
        response = httpx.get(f"{API_URL}{path}", params=dict(params), timeout=300)
    except httpx.HTTPError as exc:
        raise APIError(f"FastAPI backend is unavailable at {API_URL}: {exc}") from exc
    if response.is_error:
        raise APIError(_error_message(response))
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def api_render_page(paper_id: str, page: int, payload_json: str) -> bytes:
    try:
        response = httpx.post(
            f"{API_URL}/papers/{paper_id}/pdf/pages/{page}/render",
            content=payload_json,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
    except httpx.HTTPError as exc:
        raise APIError(f"Could not render PDF page: {exc}") from exc
    if response.is_error:
        raise APIError(_error_message(response))
    return response.content


def paper_label(paper: dict) -> str:
    year = paper.get("year") or "n.d."
    return f"{paper.get('title') or 'Untitled'}  ·  {year}"


def pair_title(pair: dict) -> str:
    return (
        f"Pair {pair['rank']} · rerank {pair['reranking_score']:.3f} · "
        f"BGE {pair['bge_similarity_score']:.3f}"
    )


def sentence_card(pair: dict, color: str) -> None:
    query = pair["query_sentence"]
    recommended = pair["recommended_sentence"]
    st.markdown(
        f"""
        <div class="evidence-card" style="--pair-color:{color}">
          <strong>{html.escape(pair_title(pair))}</strong>
          <div class="score">Query · {html.escape(str(query.get("section_heading") or "Unheaded"))}
          · page {query["page_start"]}</div>
          <div>{html.escape(query["text"])}</div><br/>
          <div class="score">Recommended · {html.escape(str(recommended.get("section_heading") or "Unheaded"))}
          · page {recommended["page_start"]}</div>
          <div>{html.escape(recommended["text"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chunk_rows(pairs: list[dict], reranked: bool) -> None:
    for pair in pairs:
        query = pair["query_chunk"]
        recommended = pair["recommended_chunk"]
        scores = f"BGE {pair['similarity_score']:.4f}"
        if reranked and pair.get("reranking_score") is not None:
            scores += f" · MiniLM {pair['reranking_score']:.4f}"
        st.markdown(f"**#{pair['rank']} · {scores}**")
        left, right = st.columns(2)
        with left:
            st.caption(
                f"Query · {query.get('section_heading') or 'Unheaded'} · "
                f"pages {query.get('page_start')}–{query.get('page_end')}"
            )
            st.write(query["text"])
        with right:
            st.caption(
                f"Recommended · {recommended.get('section_heading') or 'Unheaded'} · "
                f"pages {recommended.get('page_start')}–{recommended.get('page_end')}"
            )
            st.write(recommended["text"])
        st.divider()


def sentence_rows(pairs: list[dict], reranked: bool) -> None:
    for pair in pairs:
        score = f"BGE {pair['bge_similarity_score']:.4f}"
        if reranked and pair.get("reranking_score") is not None:
            score += f" · MiniLM {pair['reranking_score']:.4f}"
        st.markdown(f"**#{pair['rank']} · {score}**")
        st.write(pair["query_sentence"]["text"])
        st.write("↔")
        st.write(pair["recommended_sentence"]["text"])
        st.caption(
            f"Parent chunk pair #{pair['parent_chunk_pair_rank']} · "
            f"pages {pair['query_sentence']['page_start']} / "
            f"{pair['recommended_sentence']['page_start']}"
        )
        st.divider()


def highlights_for(
    pairs: list[dict], side: str, page: int, selected_index: int, show_all: bool
) -> list[dict]:
    highlights = []
    for index, pair in enumerate(pairs):
        if not show_all and index != selected_index:
            continue
        sentence = pair[f"{side}_sentence"]
        for region in sentence["coordinates"]:
            if region["page"] == page:
                highlights.append(
                    {"bbox": region["bbox"], "color": PAIR_COLORS[index % len(PAIR_COLORS)]}
                )
    return highlights


st.markdown(
    """
    <div class="hero">
      <h1>Related Work Research Assistant</h1>
      <p>Paper recommendations, passage evidence, sentence-level matching, and source-grounded PDF highlights.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Research context")
    st.caption(f"Backend: {API_URL}")
    try:
        health = api_get("/health")
        st.success(f"API: {health['status']}")
    except APIError as exc:
        st.error(str(exc))
        st.stop()

try:
    papers = api_get("/papers", (("has_embedding", True), ("limit", 500)))
except APIError as exc:
    st.error(str(exc))
    st.stop()

if not papers:
    st.warning("No papers with SPECTER2 embeddings are available.")
    st.stop()

selected_paper = st.selectbox(
    "Choose a source paper",
    papers,
    format_func=paper_label,
    key="selected_source_paper",
)
query_id = selected_paper["paper_id"]

related_work_panel(query_id)
generation_evaluation_panel(query_id)

try:
    recommendations = api_get(
        f"/papers/{query_id}/recommendations",
        (("top_k", 5), ("chunk_pairs_per_paper", 0), ("rerank", False)),
    )
except APIError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Top-5 related papers")
if not recommendations:
    st.info("No recommendations are available for the selected paper.")
    st.stop()

selected_index = st.radio(
    "Select a recommendation to inspect",
    range(len(recommendations)),
    format_func=lambda index: (
        f"#{recommendations[index]['rank']}  {recommendations[index]['title']}  "
        f"· SPECTER2 {recommendations[index]['similarity_score']:.4f}"
    ),
    key=f"recommendation_{query_id}",
)
recommendation = recommendations[selected_index]
recommended_id = recommendation["paper_id"]

with st.spinner("Matching and reranking sentence evidence…"):
    try:
        evidence = api_get(
            f"/papers/{query_id}/recommendations/{recommended_id}/evidence",
            (
                ("chunk_candidates", 20),
                ("chunk_pairs", 3),
                ("sentence_candidates", 20),
                ("sentence_pairs", 3),
            ),
        )
    except APIError as exc:
        st.error(str(exc))
        st.stop()

for warning in evidence["warnings"]:
    st.warning(warning)

sentence_pairs = evidence["sentence_pairs"]
st.subheader("Best sentence evidence")
if not sentence_pairs:
    st.info("No sentence pairs with reliable PDF mappings were found.")
else:
    for index, pair in enumerate(sentence_pairs):
        sentence_card(pair, PAIR_COLORS[index % len(PAIR_COLORS)])

    active_pair_index = st.radio(
        "Evidence pair shown in the PDF viewers",
        range(len(sentence_pairs)),
        format_func=lambda index: pair_title(sentence_pairs[index]),
        horizontal=True,
        key=f"sentence_pair_{query_id}_{recommended_id}",
    )
    show_all = st.toggle("Show all pair highlights on the selected pages", value=False)

    st.markdown("**Highlight legend**")
    for index, pair in enumerate(sentence_pairs):
        color = PAIR_COLORS[index % len(PAIR_COLORS)]
        st.markdown(
            f"<span class='legend-dot' style='background:{color}'></span>"
            f"**Pair {pair['rank']}** · MiniLM `{pair['reranking_score']:.3f}` — "
            f"{html.escape(pair['query_sentence']['text'][:110])} ↔ "
            f"{html.escape(pair['recommended_sentence']['text'][:110])}",
            unsafe_allow_html=True,
        )

    active_pair = sentence_pairs[active_pair_index]
    query_page = active_pair["query_sentence"]["page_start"]
    recommended_page = active_pair["recommended_sentence"]["page_start"]
    query_payload = json.dumps(
        {
            "highlights": highlights_for(
                sentence_pairs, "query", query_page, active_pair_index, show_all
            ),
            "zoom": 1.6,
        },
        sort_keys=True,
    )
    recommended_payload = json.dumps(
        {
            "highlights": highlights_for(
                sentence_pairs,
                "recommended",
                recommended_page,
                active_pair_index,
                show_all,
            ),
            "zoom": 1.6,
        },
        sort_keys=True,
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(f"#### Source PDF · page {query_page}")
        st.caption(selected_paper.get("title") or "Source paper")
        try:
            st.image(api_render_page(query_id, query_page, query_payload), width="stretch")
        except APIError as exc:
            st.error(str(exc))
        st.link_button("Open original source PDF", f"{API_URL}/papers/{query_id}/pdf")
    with right:
        st.markdown(f"#### Recommended PDF · page {recommended_page}")
        st.caption(recommendation.get("title") or "Recommended paper")
        try:
            st.image(
                api_render_page(recommended_id, recommended_page, recommended_payload),
                width="stretch",
            )
        except APIError as exc:
            st.error(str(exc))
        st.link_button("Open original recommended PDF", f"{API_URL}/papers/{recommended_id}/pdf")

with st.expander("SPECTER2 paper similarity and metadata"):
    st.metric("Paper similarity", f"{recommendation['similarity_score']:.4f}")
    st.json(
        {
            "rank": recommendation["rank"],
            "title": recommendation["title"],
            "authors": recommendation["authors"],
            "year": recommendation["year"],
            "venue": recommendation["venue"],
            "doi": recommendation["doi"],
        }
    )

with st.expander("BGE chunk candidates"):
    chunk_rows(evidence["bge_chunk_candidates"], reranked=False)

with st.expander("MiniLM reranked chunks"):
    chunk_rows(evidence["reranked_chunk_pairs"], reranked=True)

with st.expander("BGE sentence candidates"):
    sentence_rows(evidence["bge_sentence_candidates"], reranked=False)

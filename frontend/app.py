"""Streamlit presentation; the API owns retrieval and reasoning."""
import json
import os
import re

import requests
import streamlit as st
from dotenv import load_dotenv
from frontend.library_ui import collection_picker, evidence_card, source_url

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="Research Synthesis", page_icon="📚", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1100px; padding-top: 2.5rem;}
h1 {letter-spacing: -.035em;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
[data-testid="stTabs"] {margin-top: 1.4rem;}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Research workspace")
    st.caption("Evidence from your indexed papers")
    collection_id, collection_name = collection_picker(API_URL)
    st.caption("Questions search only the selected collection.")
    st.divider()
    with st.expander("Search settings"):
        mode_label = st.selectbox("Reasoning mode", ["Efficient", "Research baseline"])
        mode = "efficient" if mode_label == "Efficient" else "baseline"
        st.caption("Efficient: usually 2 model calls; one optional follow-up. JSON repairs may add calls." if mode == "efficient"
                   else "Original research pipeline: more model calls and a separate audit after revision.")
        use_cache = st.checkbox("Reuse recent identical answers", value=True, disabled=mode == "baseline")
        hops = st.slider("Maximum retrieval hops", 1, 2 if mode == "efficient" else 3, 2)
        top_k = st.slider("Chunks per query", 1, 4 if mode == "efficient" else 8, 4)
        audited = st.checkbox("Audit claims and revise once if needed", value=True)
    st.caption("The model's critique is a second check, not a guarantee of correctness.")

st.title("Research Synthesis")
st.caption("Ask across papers. Follow the evidence. Inspect the review.")
with st.form("question"):
    question = st.text_area("Research question", height=100, max_chars=2000,
                            placeholder=("How do IRCoT and Self-RAG decide when to retrieve evidence?" if collection_id == "default"
                                         else "What do these papers agree on, and where do their findings differ?"))
    submitted = st.form_submit_button("Synthesize", type="primary", use_container_width=True)

if submitted:
    st.session_state.pop("result", None)
    if len(question.strip()) < 5:
        st.error("Enter a research question of at least five characters.")
    else:
        with st.spinner("Searching the corpus and composing your answer…"):
            try:
                response = requests.post(API_URL + "/ask",
                                         json={"question": question.strip(), "mode": mode, "use_cache": use_cache, "max_hops": hops,
                                               "top_k": top_k, "critique": audited, "collection_id": collection_id}, timeout=(10, 600))
                if response.ok:
                    st.session_state.result = response.json()
                else:
                    st.error("The synthesis could not be completed. Check the details below and try again.")
                    with st.expander("Request details"):
                        st.text(f"HTTP {response.status_code}: {response.text[:500]}")
            except requests.RequestException:
                st.error("The research API is unavailable or timed out. Make sure the FastAPI server is running, then try again.")


def numbered_answer(answer: dict, fallback: str, numbers: dict) -> None:
    if answer.get("abstained"):
        st.info("The retrieved evidence is insufficient to answer this question.")
    elif "claims" in answer:
        for claim in answer["claims"]:
            refs = list(dict.fromkeys(numbers.get(ref, "unresolved") for ref in claim["citation_ids"]))
            st.markdown(claim["text"] + " " + " ".join(f"**[{ref}]**" for ref in refs))
    else:  # Older exported result compatibility.
        st.markdown(re.sub(r"\[(c[^\]]+)\]", lambda m: f"[{numbers.get(m[1], 'unresolved')}]", fallback))


if "result" not in st.session_state:
    st.markdown("#### Start with a focused comparison")
    st.markdown("*How do IRCoT and Self-RAG decide when to retrieve evidence?*" if collection_id == "default"
                else "*How do the methods and findings in these papers compare?*")
    st.caption("You’ll get a cited answer, source excerpts, the draft and critique, and a trace of each retrieval hop.")
    st.caption(f"Searching: {collection_name}. Add papers through Manage papers in the sidebar.")
else:
    result = st.session_state.result
    # Give every retrieved paper a stable number, including draft-only citations.
    sources = result.get("evidence", result["citations"])
    papers, numbers = {}, {}
    for item in sources:
        if item["paper_id"] not in papers:
            papers[item["paper_id"]] = len(papers) + 1
        numbers[item["chunk_id"]] = papers[item["paper_id"]]
    st.divider()
    st.subheader(result["question"])
    result_collection = result.get("collection_id", "default")
    st.caption(f"Collection: {collection_name}")
    for warning in result["warnings"]:
        st.warning(warning)
    usage = result.get("usage", {})
    if result.get("cache_hit"):
        st.caption("Reused a cached answer · 0 new model calls · 0 new inference tokens")
    elif usage:
        counts = (f"{usage['prompt_tokens']:,} input + {usage['completion_tokens']:,} output tokens"
                  if usage.get("usage_complete") else "Token totals unavailable for one or more calls")
        st.caption(f"{result.get('mode', 'baseline').capitalize()} · {usage['model_calls']} model calls · {counts}")
    cols = st.columns(3)
    cols[0].metric("Cited papers", len({c["paper_id"] for c in result["citations"]}))
    cols[1].metric("Retrieval hops", len(result["hop_log"]))
    cols[2].metric("Review", "Revised once" if result.get("revised") else
                   "Draft checked" if result["critique_log"] else "Not run")
    answer_tab, sources_tab, review_tab, retrieval_tab = st.tabs(["Answer", "Sources", "Review", "Retrieval"])
    with answer_tab:
        numbered_answer(result["final"], result["final_answer"], numbers)
        evidence_by_id = {e["chunk_id"]: e for e in sources}
        for index, claim in enumerate(result["final"].get("claims", [])):
            with st.expander(f"Explore evidence · Claim {index + 1}: {claim['text'][:90]}"):
                st.write(claim['text'])
                for ref in claim["citation_ids"]:
                    if ref not in evidence_by_id:
                        st.warning(f"Unresolved citation: {ref}")
                    else:
                        evidence_card(evidence_by_id[ref], result_collection, API_URL,
                                      f"claim_{index}_{ref}_{result_collection}")
        for limitation in result["final"]["limitations"]:
            st.info(limitation)
        st.caption("Bracketed numbers refer to papers in Sources. Check the excerpts before relying on a claim.")
    with sources_tab:
        if not sources:
            st.info("No source excerpts were retrieved.")
        cited_ids = {item["chunk_id"] for item in result["citations"]}
        for paper_id, number in papers.items():
            excerpts = [item for item in sources if item["paper_id"] == paper_id]
            st.markdown(f"**[{number}] {excerpts[0]['title']}**")
            link = source_url(excerpts[0]["url"], API_URL)
            if link:
                st.link_button("Open paper", link)
            for item in excerpts:
                label = (f"{'Cited' if item['chunk_id'] in cited_ids else 'Retrieved'} · "
                         f"PDF pages {item['page_start']}–{item['page_end']} · {item['section']}")
                with st.expander(label):
                    evidence_card(item, result_collection, API_URL, f"source_{item['chunk_id']}_{result_collection}")
    with review_tab:
        st.caption("The same model checks each claim against retrieved evidence. This is an internal audit, not an independently measured support score.")
        with st.expander("Original draft"):
            numbered_answer(result.get("draft", {}), result["draft_answer"], numbers)
        if not result["critique_log"]:
            st.info("No model review was run for this answer.")
        for index, critique in enumerate(result["critique_log"]):
            st.markdown(f"#### {'Draft' if index == 0 else 'Revision'} review")
            if not critique["verdicts"]:
                st.caption("No claims to review: the answer abstained.")
            for verdict in critique["verdicts"]:
                with st.expander(f"{verdict['claim_id']} · {verdict['status'].capitalize()}",
                                 expanded=verdict["status"] != "supported"):
                    st.write(verdict.get("explanation", ""))
                    st.caption("Evidence: " + ", ".join(verdict.get("evidence_ids", [])))
    with retrieval_tab:
        st.markdown("#### Questions searched")
        for subquestion in result["subquestions"]:
            st.markdown(f"- {subquestion}")
        for hop in result["hop_log"]:
            with st.expander(f"Hop {hop['hop']} · {len(hop['added_chunk_ids'])} new excerpts"):
                for query in hop["queries"]:
                    st.write(query)
                st.write(hop["decision"]["reason"])
                if hop["decision"].get("missing_fact"):
                    st.warning("Missing fact: " + hop["decision"]["missing_fact"])
                if hop.get("stop_reason"):
                    st.caption("Stopped: " + hop["stop_reason"].replace("_", " "))
    st.divider()
    st.download_button("Download research record", data=json.dumps(result, indent=2),
                       file_name="research-result.json", mime="application/json")

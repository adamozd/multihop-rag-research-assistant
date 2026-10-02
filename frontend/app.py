import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Research Synthesis", page_icon="📚", layout="wide")
st.title("Research Synthesis")
st.caption("Multi-paper answers · iterative retrieval · claim-by-claim critique")
with st.form("question"):
    question = st.text_area("Research question", placeholder="How do IRCoT and Self-RAG decide when to retrieve evidence?")
    left, right = st.columns(2)
    hops = left.slider("Maximum retrieval hops", 1, 3, 2)
    top_k = right.slider("Chunks per query", 1, 8, 4)
    audited = st.checkbox("Audit claims and revise once if needed", value=True)
    submitted = st.form_submit_button("Synthesize", type="primary")

if submitted:
    st.session_state.pop("result", None)
    if len(question.strip()) < 5:
        st.error("Enter a research question of at least five characters.")
    else:
        with st.spinner("Retrieving papers, synthesizing claims, and checking evidence…"):
            try:
                response = requests.post(os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/") + "/ask",
                                         json={"question": question, "max_hops": hops, "top_k": top_k, "critique": audited},
                                         timeout=(10, 600))
                if response.ok:
                    st.session_state.result = response.json()
                else:
                    st.error(f"Request failed ({response.status_code}): {response.text[:500]}")
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")

if "result" in st.session_state:
    result = st.session_state.result
    st.caption(result["question"])
    for warning in result["warnings"]:
        st.warning(warning)
    st.subheader("Final answer")
    st.write(result["final_answer"])
    for limitation in result["final"]["limitations"]:
        st.info(limitation)
    with st.expander("Draft answer"):
        st.write(result["draft_answer"])
    with st.expander("Citations and source excerpts", expanded=True):
        for item in result["citations"]:
            st.markdown(f"**{item['chunk_id']}** · [{item['title']}]({item['url']})")
            st.caption(f"{item['section']} · PDF pages {item['page_start']}–{item['page_end']}")
            st.text(item["text"])
    with st.expander("Critique log"):
        if not result["critique_log"]:
            st.write("Critique was disabled.")
        for index, critique in enumerate(result["critique_log"]):
            st.markdown(f"**{'Draft' if index == 0 else 'Revision'} audit**")
            st.dataframe(critique["verdicts"], use_container_width=True)
    with st.expander("Retrieval trace"):
        st.json({"subquestions": result["subquestions"], "hops": result["hop_log"]})
    st.download_button("Download full result", data=__import__("json").dumps(result, indent=2),
                       file_name="research-result.json", mime="application/json")

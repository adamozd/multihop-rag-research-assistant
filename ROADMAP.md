# Project notes

This project started with a simple question: can a research assistant give better answers by checking its own claims against the papers it retrieved?

The app now runs from question to cited answer, including a critique and one revision when needed. Recent work improved PDF extraction, kept searches more focused, and made the sources and review history easier to explore in Streamlit.

The first live evaluation on IBM watsonx completed 11 of 18 questions before the token quota ran out. The three separate acceptance checks were also blocked. The saved results have since been reviewed with AI assistance, but the partial run does not establish whether the critique loop improves answer quality. The [evaluation summary](eval/reports/README.md) has the results and limitations.

There are still a couple of rough edges: retrieval can find the right paper but miss the passage that answers the question, and revision sometimes leaves unsupported claims in the answer.

Evaluation work is paused for now. The app now supports PDF uploads into separate research collections and a claim-by-claim evidence explorer with original PDF pages. After exhausting the free quota, I added a smaller default flow, cached repeat answers, usage tracking and request limits. The original flow is still available for research comparisons. The next step is to check the smaller flow against real evidence when quota is available; its answer quality has not yet been measured.

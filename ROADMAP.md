# Project notes

This project started with a simple question: can a research assistant give better answers by checking its own claims against the papers it retrieved?

The app now runs from question to cited answer, including a critique and one revision when needed. Recent work improved PDF extraction, kept searches more focused, and made the sources and review history easier to explore in Streamlit.

The first live evaluation on IBM watsonx completed 11 of 18 questions before the token quota ran out. The three separate acceptance checks were also blocked. The saved results are ready for human review, so it's too early to say whether the critique loop improves answer quality. The [evaluation summary](eval/reports/README.md) has the results and limitations.

There are still a couple of rough edges: retrieval can find the right paper but miss the passage that answers the question, and revision sometimes leaves unsupported claims in the answer.

The immediate focus is to review the saved answers, add a way to resume interrupted evaluations, and finish the remaining checks when quota is available. Those results will guide what needs attention next.

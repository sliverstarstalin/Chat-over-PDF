from __future__ import annotations

import time

from qa import answer_question, retrieve, QA_PROMPT
from langchain_openai import ChatOpenAI

QUERIES = [
    "Who was Ashoka?",
    "Describe Ashoka's rule.",
    "Summarize the document."
]


def main() -> None:
    llm = ChatOpenAI()
    for q in QUERIES:
        ctx_chunks = retrieve(q)
        context = "\n".join(c.text for c in ctx_chunks)
        prompt = QA_PROMPT.format(context=context, question=q)
        prompt_tokens = llm.get_num_tokens(prompt)

        start = time.perf_counter()
        ans = answer_question(q)
        elapsed = time.perf_counter() - start

        completion_tokens = llm.get_num_tokens(ans.text)

        print(f"Q: {q}")
        print(ans.text)
        print(sorted(ans.citations))
        print(f"latency: {elapsed:.2f}s")
        print(f"prompt: {prompt_tokens} tokens, completion: {completion_tokens} tokens")
        print()


if __name__ == "__main__":
    main()

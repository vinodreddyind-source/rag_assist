"""
LCEL -- LangChain Expression Language. This is the `prompt | llm | parser`
pipe-operator syntax you'll be asked to recognize even though we hand-rolled
most of this project's logic in plain Python.

Why we hand-rolled instead of using LCEL throughout: LCEL is excellent for
LINEAR chains, but this project's whole point (in Phase 2) is a graph with
loops and conditional branching -- LangGraph, not LCEL, is the right tool
for that, and your Guidewire doc already has the crisp answer for "why
LangGraph instead of LangChain" (section 3). LCEL is worth demonstrating
because interviewers may ask "have you used LCEL specifically" as a check
that you know the ecosystem beyond just LangGraph.

The `|` operator overload works because every LCEL component implements a
common Runnable interface (invoke/batch/stream/ainvoke). Each stage takes
the previous stage's output as its input -- this file shows the equivalent
of our generate.py's prompt-formatting + LLM call, written as an LCEL chain
against a local Ollama model instead of raw requests.post().
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


def format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        meta = c["metadata"]
        parts.append(f"[Source: {meta['product']} / {meta['section']}]\n{c['text']}")
    return "\n\n".join(parts)


def build_lcel_chain():
    """RUN ON YOUR LAPTOP -- needs `langchain-ollama` and a running Ollama.

    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.1:8b")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using ONLY the provided context. Cite sources."),
        ("human", "Context:\\n{context}\\n\\nQuestion: {query}"),
    ])

    # This is the LCEL chain: each stage's output becomes the next stage's input
    chain = prompt | llm | StrOutputParser()

    # Usage:
    #   chain.invoke({"context": format_context(chunks), "query": query})
    #   chain.stream({"context": ..., "query": ...})   # token-by-token, for TTFT
    #   chain.batch([{"context": c1, "query": q1}, {"context": c2, "query": q2}])

    return chain
    """
    raise NotImplementedError(
        "This needs a real Ollama connection -- see the docstring above for "
        "the exact 4 lines to run on your laptop. The point of this file is "
        "the docstring: the pipe-chain shape, and *why* it doesn't fit "
        "Phase 2's retry loop (that's LangGraph's job, not LCEL's)."
    )


# A pure-Python illustration of the SAME composition idea, runnable here with
# no external model -- so you can see what `|` is actually doing under the
# hood before trusting the real LCEL version on your laptop.
def demo_composable_runnables():
    step1 = RunnableLambda(lambda x: x.upper())
    step2 = RunnableLambda(lambda x: f"[PROCESSED] {x}")
    chain = step1 | step2   # <-- same pipe operator, same Runnable interface
    return chain.invoke("hello world")


if __name__ == "__main__":
    print("LCEL Runnable composition (no LLM needed to demonstrate the pattern):")
    print(demo_composable_runnables())

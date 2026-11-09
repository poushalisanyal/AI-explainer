"""
D_agents_v3.py - the AI agent, built as a small LangGraph graph.

              question
                 |
          [search_books]            OKF exact section match, else
                 |                  vector search in ChromaDB      (B_rag.py)
                 |
     is the best match good enough?
           yes /     \\ no
              /       \\
 [answer_from_books]  [answer_from_knowledge]                       (C_tool.py)
              \\       /
                 END

Test:  python D_agents_v3.py
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from B_rag import MIN_SCORE, BookSearch
from C_tool import explain

books = BookSearch()


class State(TypedDict, total=False):
    question: str
    level: str
    history: list          # earlier messages: [("user", "..."), ("assistant", "..."), ...]
    passages: list         # book chunks found by search_books
    answer: dict           # final answer


# ── Graph nodes ───────────────────────────────────────────────────────────────

def search_books(state):
    query = state["question"]
    # Short follow-ups like "give an example" need the previous question to search well
    if len(query.split()) < 6 and state["history"]:
        previous_question = state["history"][-2][1]
        query = f"{previous_question} {query}"
    return {"passages": books.search(query)}


def choose_path(state):
    best_score = state["passages"][0]["score"] if state["passages"] else 0
    return "books" if best_score >= MIN_SCORE else "knowledge"


def answer_from_books(state):
    good = [p for p in state["passages"] if p["score"] >= MIN_SCORE]
    passages = "\n\n".join(f"[{p['book']}, page {p['page']}]\n{p['text']}" for p in good)
    related = []
    for p in good:
        related += [topic for topic in p["related"] if topic not in related]
    answer = explain(state["question"], state["level"], state["history"], passages, related[:6])

    sources = []
    for p in good:
        source = f"{p['book']}, p. {p['page']} ({p['section']})"
        if source not in sources:
            sources.append(source)
    answer["sources"] = sources
    answer["found_by"] = "exact section (OKF)" if good[0]["found_by"] == "okf" else "semantic search"
    return {"answer": answer}


def answer_from_knowledge(state):
    answer = explain(state["question"], state["level"], state["history"])
    answer["sources"] = []
    answer["found_by"] = "general knowledge"
    return {"answer": answer}


# ── Build the graph ───────────────────────────────────────────────────────────

graph = StateGraph(State)
graph.add_node("search_books", search_books)
graph.add_node("answer_from_books", answer_from_books)
graph.add_node("answer_from_knowledge", answer_from_knowledge)

graph.set_entry_point("search_books")
graph.add_conditional_edges("search_books", choose_path,
                            {"books": "answer_from_books", "knowledge": "answer_from_knowledge"})
graph.add_edge("answer_from_books", END)
graph.add_edge("answer_from_knowledge", END)

agent = graph.compile()


# ── Public function used by main.py ───────────────────────────────────────────

OUT_OF_SCOPE = {
    "is_ai_question": False,
    "term": "",
    "summary": "",
    "explanation": "I can only explain AI, machine learning and the maths behind them. "
                   "Try one of the questions below.",
    "analogy": "", "formula": "", "example": "", "common_mistake": "",
    "follow_up_questions": ["What is a neural network?", "What is gradient descent?",
                            "How does a transformer work?"],
    "sources": [],
    "found_by": "",
}

cache = {}   # same question + same level = same answer, so we save Groq requests


def ask(question, level="beginner", history=None):
    history = history or []
    cache_key = (question.lower().strip(" ?!."), level)
    if not history and cache_key in cache:
        return cache[cache_key]

    result = agent.invoke({"question": question, "level": level, "history": history})
    answer = result["answer"]
    if not answer["is_ai_question"]:
        return OUT_OF_SCOPE

    if not history:
        cache[cache_key] = answer
    return answer


if __name__ == "__main__":
    import json
    print(json.dumps(ask("What is backpropagation?"), indent=2))
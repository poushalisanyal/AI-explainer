"""
main.py - the web server (FastAPI).

Run:   uvicorn main:app --reload
Open:  http://127.0.0.1:8000
"""

from typing import Literal

import groq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import E_database as db
from D_agents_v3 import ask

app = FastAPI(title="AI Concepts Explainer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

Level = Literal["beginner", "intermediate", "advanced"]


class NewConversationRequest(BaseModel):
    level: Level = "beginner"


class ChatRequest(BaseModel):
    conversation_id: str
    question: str
    level: Level = "beginner"


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/conversation/new")
def new_conversation(request: NewConversationRequest):
    return {"conversation_id": db.create_conversation(request.level), "level": request.level}


@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please type a question.")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Please keep the question under 1000 characters.")
    if not db.conversation_exists(request.conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")

    history = db.get_history(request.conversation_id)
    try:
        answer = ask(question, request.level, history)
    except groq.RateLimitError:
        raise HTTPException(status_code=429, detail="The free Groq limit was reached. Wait a minute and try again.")

    db.save_turn(request.conversation_id, question, answer)
    return {**answer, "conversation_id": request.conversation_id, "question": question}


@app.get("/conversation/{conversation_id}")
def get_conversation(conversation_id: str):
    if not db.conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"conversation_id": conversation_id, "messages": db.get_messages(conversation_id)}


@app.get("/health")
def health():
    from D_agents_v3 import books
    return {"status": "ok",
            "okf_sections": len(books.okf.sections),
            "okf_terms": len(books.okf.by_alias),
            "vector_chunks": books.collection.count()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
"""
E_database.py - saves conversations and messages with SQLAlchemy.

Uses SQLite by default (a file called ai_explainer.db).
For PostgreSQL, set DATABASE_URL in .env, e.g.
    DATABASE_URL=postgresql+psycopg://user:password@host/dbname
"""

import json
import os
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_explainer.db")
HISTORY_TURNS = 3   # how many earlier questions + answers are sent to the LLM

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
Session = sessionmaker(bind=engine)
Base = declarative_base()


def now():
    return datetime.now(timezone.utc)


# ── Tables ────────────────────────────────────────────────────────────────────

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True)
    level = Column(String, default="beginner")
    created_at = Column(DateTime, default=now)


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String, nullable=False)       # "user" or "assistant"
    content = Column(Text, nullable=False)      # question text, or the answer as JSON
    created_at = Column(DateTime, default=now)


Base.metadata.create_all(engine)   # creates the tables the first time


# ── Functions used by main.py ─────────────────────────────────────────────────

def create_conversation(level="beginner"):
    conversation_id = str(uuid.uuid4())
    with Session() as session:
        session.add(Conversation(id=conversation_id, level=level))
        session.commit()
    return conversation_id


def conversation_exists(conversation_id):
    with Session() as session:
        return session.get(Conversation, conversation_id) is not None


def save_turn(conversation_id, question, answer):
    """Save the user's question and the agent's answer."""
    with Session() as session:
        session.add(Message(conversation_id=conversation_id, role="user", content=question))
        session.add(Message(conversation_id=conversation_id, role="assistant", content=json.dumps(answer)))
        session.commit()


def get_messages(conversation_id):
    """All messages of a conversation, oldest first (answers turned back into dicts)."""
    with Session() as session:
        rows = (session.query(Message)
                .filter(Message.conversation_id == conversation_id)
                .order_by(Message.id)
                .all())
        return [{"role": m.role,
                 "content": json.loads(m.content) if m.role == "assistant" else m.content,
                 "time": m.created_at.isoformat()}
                for m in rows]


def get_history(conversation_id):
    """The last few turns in the format the LLM needs: [("user", "..."), ("assistant", "...")]."""
    history = []
    for m in get_messages(conversation_id)[-HISTORY_TURNS * 2:]:
        if m["role"] == "user":
            history.append(("user", m["content"]))
        else:
            history.append(("assistant", m["content"].get("explanation", "")))
    return history
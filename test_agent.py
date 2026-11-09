"""
test_agent.py - asks every question in questions.json and saves the answers.

Start the server first (uvicorn main:app), then in another terminal:
    python test_agent.py                    # level picked per question category
    python test_agent.py --level beginner   # same level for every question
    python test_agent.py --fresh            # start over instead of resuming

Results go to answers.json and answers.csv (open it in Excel).
If it stops, run it again: it continues where it left off.
"""

import argparse
import csv
import json
import os
import sys
import time

import requests

BASE_URL = "http://127.0.0.1:8000"

LEVEL_BY_CATEGORY = {
    "concept_basic": "beginner",
    "concept_intermediate": "intermediate",
    "concept_advanced": "advanced",
    "calculation": "intermediate",
}


def readable_answer(answer):
    """Turn the JSON answer into one block of text for the CSV file."""
    parts = []
    for field in ["term", "summary", "explanation", "formula", "analogy", "example", "common_mistake"]:
        if answer.get(field):
            parts.append(f"[{field}] {answer[field]}")
    if answer.get("found_by"):
        parts.append(f"[found_by] {answer['found_by']}")
    if answer.get("sources"):
        parts.append("[sources] " + " | ".join(answer["sources"]))
    if answer.get("follow_up_questions"):
        parts.append("[follow_ups] " + " | ".join(answer["follow_up_questions"]))
    return "\n".join(parts)


def ask(base, question, level):
    """Each question gets its own new conversation, so answers don't affect each other."""
    conversation = requests.post(f"{base}/conversation/new", json={"level": level}, timeout=30).json()
    return requests.post(f"{base}/chat", timeout=180, json={
        "conversation_id": conversation["conversation_id"], "question": question, "level": level})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=BASE_URL)
    parser.add_argument("--level", default="auto", choices=["auto", "beginner", "intermediate", "advanced"])
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--delay", type=float, default=8, help="seconds between questions (Groq free limit)")
    args = parser.parse_args()

    try:
        requests.get(f"{args.base}/health", timeout=10)
    except Exception:
        sys.exit(f"Can't reach {args.base}. Is the server running?")

    with open("questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    if os.path.exists("answers.json") and not args.fresh:
        with open("answers.json", encoding="utf-8") as f:
            results = [r for r in json.load(f) if r["status"] not in ("error", "http_429")]
    done = {r["id"] for r in results}

    for q in questions:
        if q["id"] in done:
            continue
        level = args.level if args.level != "auto" else LEVEL_BY_CATEGORY.get(q["category"], "beginner")
        print(f"[{q['id']}/{len(questions)}] ({level}) {q['question'][:65]}")

        start = time.time()
        row = {"id": q["id"], "category": q["category"], "level": level, "question": q["question"],
               "expected": q.get("expected_behavior", "")}
        try:
            response = ask(args.base, q["question"], level)
            data = response.json()
            if response.ok:
                row["status"] = "answer" if data.get("is_ai_question") else "out_of_scope"
                row["answer"] = readable_answer(data)
            else:
                row["status"] = f"http_{response.status_code}"
                row["answer"] = data.get("detail", "")
        except Exception as e:
            row["status"] = "error"
            row["answer"] = str(e)
        row["seconds"] = round(time.time() - start, 1)

        results.append(row)
        with open("answers.json", "w", encoding="utf-8") as f:
            json.dump(sorted(results, key=lambda r: r["id"]), f, indent=2, ensure_ascii=False)
        time.sleep(args.delay)

    with open("answers.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "level", "question", "expected",
                                               "status", "answer", "seconds"])
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: r["id"]))
    print(f"Done: {len(results)} answers saved to answers.json and answers.csv")


if __name__ == "__main__":
    main()
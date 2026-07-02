# test_rag_manual.py
# Run from backend/ folder: python test_rag_manual.py
# Tests the full RAG pipeline: ChromaDB retrieval + Groq LLM answer

import sys
import json
sys.path.insert(0, '.')
import rag_pipeline

print("=" * 60)
print("TEST 1 — Signal question with department filter")
print("=" * 60)
result = rag_pipeline.answer_question(
    question="What should I do if the axle counter fails?",
    department="signal"
)
print("ANSWER:\n", result["answer"])
print("\nSOURCES:")
for s in result["sources"]:
    print(f"  • {s['filename']}  page {s['page']}")
print(f"\nChunks used: {result['chunks_used']}")
print(f"Department searched: {result['department']}")

print("\n" + "=" * 60)
print("TEST 2 — Operations question, no department filter")
print("=" * 60)
result2 = rag_pipeline.answer_question(
    question="What are the steps to follow during a train accident?",
    department=None
)
print("ANSWER:\n", result2["answer"])
print("\nSOURCES:")
for s in result2["sources"]:
    print(f"  • {s['filename']}  page {s['page']}")

print("\n" + "=" * 60)
print("TEST 3 — Invalid department (should return clean error)")
print("=" * 60)
result3 = rag_pipeline.answer_question(
    question="anything",
    department="nonexistent"
)
print("ANSWER:", result3["answer"])
print("Error key:", result3.get("error"))
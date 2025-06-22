import json
from transformers import pipeline

# Load model again
qa_model = pipeline("text-generation", model="distilgpt2", max_new_tokens=20)

# Load KB
with open("kb.json", "r") as f:
    kb = json.load(f)

# Normalize text for comparison
def normalize(text):
    return text.strip().lower()

# Read run.log
with open("run.log", "r") as f:
    entries = [json.loads(line) for line in f]

# Validate each entry
for entry in entries:
    question = entry["question"]
    answer = normalize(entry["answer"])

    print(f"\nQ: {question}")
    print(f"A: {entry['answer']}")

    if question in kb:
        correct = normalize(kb[question])
        if correct in answer or answer in correct:
            print("✅ MATCHED KB")
        else:
            print("❌ RETRY: answer differs from KB")
            retry = qa_model(question)[0]['generated_text'][len(question):].strip()
            print(f"🔁 Retry Answer: {retry}")
    else:
        print("❌ RETRY: out-of-domain")
        retry = qa_model(question)[0]['generated_text'][len(question):].strip()
        print(f"🔁 Retry Answer: {retry}")

import json
from transformers import pipeline

# Load the model pipeline
qa_model = pipeline("text-generation", model="distilgpt2", max_new_tokens=20)

# Load known KB questions
with open("kb.json", "r") as f:
    kb = json.load(f)

known_questions = list(kb.keys())

# Add 5 edge/unseen questions
unseen_questions = [
    "What is the capital of Middle Earth?",
    "Who discovered time travel?",
    "How many colors are there in a unicorn's horn?",
    "What is the taste of the number five?",
    "Can AI feel love?"
]

# Combine all 15 questions
all_questions = known_questions + unseen_questions

# Ask model and log results
log = []

for q in all_questions:
    print(f"Asking: {q}")
    result = qa_model(q)[0]['generated_text']
    answer = result[len(q):].strip()
    print(f"Answer: {answer}")
    
    log.append({
        "question": q,
        "answer": answer
    })

# Save to run.log
with open("run.log", "w") as f:
    for entry in log:
        f.write(json.dumps(entry) + "\n")

print("\nAll questions asked. Answers saved to run.log ✅")

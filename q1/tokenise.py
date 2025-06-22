print("Tokenization Comparison")
from transformers import AutoTokenizer
from transformers import pipeline
import sentencepiece as spm
import os


sentence = "The cat sat on the mat because it was tired."

# Load pre-trained tokenizers
bpe_tokenizer = AutoTokenizer.from_pretrained("gpt2")  # BPE
wordpiece_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # WordPiece
sentencepiece_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")  # SentencePiece

def report(name, tokenizer):
    print(f"\n--- {name} ---")
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print("Tokens: ", tokens)
    print("Token IDs:", token_ids)
    print("Total Tokens:", len(tokens))

report("BPE (GPT-2)", bpe_tokenizer)
report("WordPiece (BERT)", wordpiece_tokenizer)
report("SentencePiece (T5)", sentencepiece_tokenizer)


# Load the masked language model pipeline
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# Sentence with two [MASK] tokens
masked_sentence = "The cat sat on the [MASK] because it was [MASK]."

# Run prediction
predictions = fill_mask(masked_sentence)

# Check structure
print("\nDEBUG: Raw predictions (list of lists):")
print(predictions)

# Handle first [MASK]
print("\n--- First [MASK] predictions ---")
first_mask_preds = predictions[0]  # first list

for i, item in enumerate(first_mask_preds[:3]):
    print(f"Prediction {i+1}: {item['sequence']} (score: {item['score']:.4f})")

# Handle second [MASK]
print("\n--- Second [MASK] predictions ---")
second_mask_preds = predictions[1]  # second list

for i, item in enumerate(second_mask_preds[:3]):
    print(f"Prediction {i+1}: {item['sequence']} (score: {item['score']:.4f})")
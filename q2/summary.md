# Model Performance Summary

## Overall Results

- **Correct Answers:** X  
- **Incorrect Answers (Mismatched):** Y  
- **Out-of-Domain Questions:** Z  
- **Total Retries:** R  

---

## Detailed Breakdown

- **Question 1:** ✅ Correct  
- **Question 2:** ❌ Incorrect (mismatched)  
- **Question 3:** ⚠️ Out-of-Domain  
- ...  

---

## Notes

- The system simulates model answers using a dummy function that returns correct answers 70% of the time for in-KB questions.
- For out-of-domain questions, the model returns a default out-of-domain message.
- The validator detects mismatches and retries the question once to get a better answer.
- This setup helps test the hallucination detection and guardrail logic without requiring a heavy AI model.
- In a real-world scenario, this could be extended to use a real language model like Mistral 7B or similar.

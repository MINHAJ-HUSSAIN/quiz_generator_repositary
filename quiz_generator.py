
import streamlit as st
from transformers import pipeline

# Load the question-answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Function to generate quiz questions and answers from input text
def generate_quiz(text):
    questions = []
    # Example questions based on general prompts
    inputs = [
        {"question": "What is the main topic of the article?", "context": text},
        {"question": "What are the key points covered?", "context": text},
        {"question": "Who is the article about?", "context": text},
        {"question": "What events are described in the text?", "context": text}
    ]
    
    # Generate answers for each question using the QA pipeline
    for inp in inputs:
        result = qa_pipeline(question=inp["question"], context=inp["context"])
        questions.append((inp["question"], result["answer"]))
    return questions

def main():
    st.title("AI-Based Quiz Generator")
    st.write("Paste an article or study material below to automatically generate quiz questions.")

    # Text input from the user
    user_text = st.text_area("Paste text here:", "")

    if st.button("Generate Quiz"):
        if user_text:
            quiz = generate_quiz(user_text)
            st.write("### Generated Quiz Questions and Answers:")
            for idx, (q, a) in enumerate(quiz):
                st.write(f"**Q{idx+1}: {q}**")
                st.write(f"**A{idx+1}: {a}**")
                st.write("---")
        else:
            st.warning("Please enter some text to generate quiz questions.")

if __name__ == "__main__":
    main()

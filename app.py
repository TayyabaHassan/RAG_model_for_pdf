import gradio as gr
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import os

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract text from each PDF
def extract_text_from_pdfs(pdf_files):
    all_texts = []
    for file in pdf_files:
        with fitz.open(file.name) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            all_texts.append(text)
    return all_texts

# Generate short preview of PDFs
def preview_pdfs(pdf_files):
    previews = []
    for file in pdf_files:
        with fitz.open(file.name) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
                if len(text) > 500:
                    break
            previews.append(f"{file.name}:\n{text[:500]}...\n")
    return "\n".join(previews)

# Simple chunk splitter
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Get embeddings
def embed_texts(texts):
    return embedding_model.encode(texts)

# Retrieve top relevant chunks with labels
def retrieve_chunks(question, chunk_embeddings, chunks, top_k=3):
    question_embedding = embedding_model.encode([question])
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(chunks[i], f"Chunk #{i + 1}") for i in top_indices]

# Call Groq LLM (llama3-8b-8192)
def call_llm_api(question, context):
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    json_data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=json_data,
            timeout=15
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"🔌 Connection error: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"

# Main answer logic
def answer_question(question, pdf_files):
    texts = extract_text_from_pdfs(pdf_files)
    chunks = []
    for text in texts:
        chunks.extend(split_text(text))
    chunk_embeddings = embed_texts(chunks)
    relevant_chunks = retrieve_chunks(question, chunk_embeddings, chunks)
    context = "\n\n".join([f"[{src}]\n{text}" for text, src in relevant_chunks])
    return call_llm_api(question, context)

# Preview UI
def show_previews(pdf_files):
    return preview_pdfs(pdf_files)

# Gradio UI
with gr.Blocks(title="RAG PDF Chatbot") as demo:
    gr.Markdown("## 📚 RAG-Based PDF Chatbot with Groq LLM")
    with gr.Row():
        pdf_input = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDFs")
    with gr.Row():
        preview_button = gr.Button("📄 Show PDF Summaries")
        preview_output = gr.Textbox(label="PDF Previews", lines=10)
    preview_button.click(fn=show_previews, inputs=[pdf_input], outputs=[preview_output])

    gr.Markdown("### ❓ Ask a Question")
    question_input = gr.Textbox(label="Your Question")
    answer_output = gr.Textbox(label="Answer", lines=10)
    ask_button = gr.Button("🔍 Get Answer")
    ask_button.click(fn=answer_question, inputs=[question_input, pdf_input], outputs=[answer_output])

# Run
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

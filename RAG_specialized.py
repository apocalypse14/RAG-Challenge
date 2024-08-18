import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np
import time
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def preprocess_markdown(markdown_file):
    with open(markdown_file, 'r', encoding='utf-8') as file:
        text = file.read()
    text = re.sub(r'\n\s*\n', '\n', text)  # remove empty lines
    text = re.sub(r'!\[.*?]\(.*?\)', '', text)  # remove images
    sections = re.split(r'(## .*)', text)
    chunks = []
    for i in range(1, len(sections), 2):
        header = sections[i].strip()
        content = sections[i + 1].strip()
        # This optimizes chunking by prohibiting too large or too small chunks
        if len(content.split()) < 100:
            if chunks:
                chunks[-1] += f"\n\n{header}\n{content}"
            else:
                chunks.append(f"{header}\n{content}")
        else:
            paragraphs = content.split("\n\n")
            for paragraph in paragraphs:
                chunks.append(f"{header}\n{paragraph.strip()}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    final_chunks = []
    for chunk in chunks:
        final_chunks.extend(splitter.split_text(chunk))
    return final_chunks


def retrieve_relevant_chunks(query, index, model, chunks, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, top_k)
    if len(indices[0]) == 0:
        print("No relevant chunks found.")
        return "I don't know."
    relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return relevant_chunks


def save_embeddings(embedding_matrix, file_path):
    np.save(file_path, embedding_matrix)
    print(f"Embeddings saved to {file_path}")


def load_embeddings(file_path):
    embedding_matrix = np.load(file_path)
    print(f"Embeddings loaded from {file_path}")
    return embedding_matrix


def clean_output(answer):
    sentences = answer.split('. ')
    cleaned_answer = '. '.join(sentences[:-1]) + '.'
    if not cleaned_answer.strip() or cleaned_answer.strip() == ".":  # unexpected output will return "Idk"
        return "I don't know."
    return cleaned_answer


def generate_answer(relevant_chunks, query):
    context = " ".join(relevant_chunks)
    prompt_template = f"""Context: {context}\nQuestion: {query}"""
    inputs = tokenizer.encode(prompt_template, return_tensors="pt").to(device)
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)
    outputs = gpt_model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=150,
        num_return_sequences=1,
        repetition_penalty=2.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return clean_output(text)


def rag_pipeline(markdown_file, query, top_k=5, embeddings_file="embeddings.npy"):
    chunks = preprocess_markdown(markdown_file)  # This returns a list of strings
    if os.path.exists(embeddings_file):
        print(f"Loading embeddings from {embeddings_file}")
        embedding_matrix = load_embeddings(embeddings_file)
    else:
        print(f"Embeddings file not found. Creating embeddings and saving to {embeddings_file}")
        embeddings = embedding_model.encode(chunks, convert_to_tensor=True).cpu().numpy()
        embedding_matrix = np.array(embeddings)
        save_embeddings(embedding_matrix, embeddings_file)
    # For Debugging: FAISS index dimension
    # print(f"Embedding matrix shape: {embedding_matrix.shape}")
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    relevant_chunks = retrieve_relevant_chunks(query, index, embedding_model, chunks, top_k)
    return generate_answer(relevant_chunks, query)


if __name__ == '__main__':
    start_time = time.time()
    device = torch.device("cpu")
    model = "EleutherAI/gpt-neo-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model)
    gpt_model = AutoModelForCausalLM.from_pretrained(model).to(device)
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    tokenizer.pad_token = tokenizer.eos_token
    markdown_file = r'C:\Users\User\PycharmProjects\CokGüzelOlacak\Data\git-tutorial.md'
    query = "how to inspect a remote?"  # Ask a question here about the git-tutorial.md
    answer = rag_pipeline(markdown_file, query)
    end_time = time.time()
    print("-----------------------------------------------------------------------------------------------------------")
    print(answer + "\n")
    print("The response took " + str(end_time - start_time) + "s.")
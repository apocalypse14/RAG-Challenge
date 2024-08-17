import os
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import faiss
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_markdown(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()

    text = re.sub(r'\n\s*\n', '\n', text)
    sections = re.split(r'(## .*)', text)

    headers = []
    section_map = {}

    for i in range(1, len(sections), 2):
        header = sections[i].strip()
        content = sections[i + 1].strip()

        # A section starts with ## and ends with a .
        content = re.split(r'(## .*)', content)[0].strip()
        headers.append(header)
        section_map[header] = content
    return headers, section_map


def find_best_matching_header(query, headers, model):
    header_embeddings = model.encode(headers, convert_to_tensor=True)
    query_embedding = model.encode([query], convert_to_tensor=True)

    similarities = cosine_similarity(query_embedding, header_embeddings).flatten()

    best_index = np.argmax(similarities) # this returns the index of the best matching header
    return headers[best_index]


def save_embeddings(embedding_matrix, file_path):
    np.save(file_path, embedding_matrix)
    print(f"Embeddings saved to {file_path}")


def load_embeddings(file_path):
    embedding_matrix = np.load(file_path)
    print(f"Embeddings loaded from {file_path}")
    return embedding_matrix


def generate_summary(context):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(context, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']


def generate_answer(context, query):
    print("Context: " + context)
    print("Question: " + query + "\n")
    summary = generate_summary(context)
    # I can also combine the summary with the query for fine-tuning, but just the summary is faster:
    # prompt_template = f"""Context: {summary}\nQuestion: {query}"""
    return clean_output(summary)


def clean_output(answer):
    sentences = answer.split('. ')
    cleaned_answer = '. '.join(sentences[:-1]) + '.'
    if not cleaned_answer.strip() or cleaned_answer.strip() == ".":  # unexpected output will return "IDK"
        return "I don't know."
    return cleaned_answer


def rag_pipeline(markdown_file, query, embeddings_file="embeddings.npy"):
    headers, section_map = preprocess_markdown(markdown_file)
    best_header = find_best_matching_header(query, headers, model)
    context = section_map[best_header]
    # Don't need for Faiss search since I only use a single context
    return generate_answer(context, query)


if __name__ == '__main__':
    start_time = time.time()
    device = torch.device("cpu")  # use cpu
    model = "EleutherAI/gpt-neo-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model)
    gpt_model = AutoModelForCausalLM.from_pretrained(model).to(device)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer.pad_token = tokenizer.eos_token

    # This part should be tailored depending on the md file used.
    markdown_file = r'C:\Users\User\PycharmProjects\CokGüzelOlacak\Data\git-tutorial.md'  # file_path

    # This is the question that will be queried on the specified file.
    query = "how do I use branches in git?"
    answer = rag_pipeline(markdown_file, query)
    end_time = time.time()
    print("Answer: " + answer)
    print("The response took " + str(end_time - start_time) + "s.")

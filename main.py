import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np
import time


def preprocess_markdown(markdown_file):
    with open(markdown_file, 'r', encoding='utf-8') as file:
        text = file.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=25,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks


def save_embeddings(embedding_matrix, file_path):
    np.save(file_path, embedding_matrix)
    print(f"Embeddings saved to {file_path}")

def load_embeddings(file_path):
    embedding_matrix = np.load(file_path)
    print(f"Embeddings loaded from {file_path}")
    return embedding_matrix


def retrieve_relevant_chunks(query, index, model, chunks, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]  # list of chunks
    return relevant_chunks


def generate_answer(relevant_chunks, query):
    context = "".join(relevant_chunks)
    prompt_template = f"""Context: {context}\nQuestion: {query}"""
    inputs = tokenizer.encode(prompt_template, return_tensors="pt").to(device)
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)

    outputs = gpt_model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=50,  # better than max_length since it input query is excluded
        # max_new_token length affects the speed of output generation.
        num_return_sequences=1,
        repetition_penalty=2.0,  # reduce repetition
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer if answer else "I don't know."


def rag_pipeline(markdown_file, query, top_k=5, embeddings_file="embeddings.npy"):
    chunks = preprocess_markdown(markdown_file)
    if os.path.exists(embeddings_file):
        embedding_matrix = load_embeddings(embeddings_file)
    else:
        emb_start_time = time.time()
        embeddings = model.encode(chunks, convert_to_tensor=True).tolist()
        embedding_matrix = np.array(embeddings)
        emb_end_time = time.time()
        print(f"Embeddings took {emb_end_time - emb_start_time:.4f} seconds.")
        save_embeddings(embedding_matrix, embeddings_file)

    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    relevant_chunks = retrieve_relevant_chunks(query, index, model, chunks, top_k)
    return generate_answer(relevant_chunks, query)


if __name__ == '__main__':
    start_time = time.time()
    device = torch.device("cpu")
    model = "EleutherAI/gpt-neo-1.3B"
    # model = "gpt2"  # this is the smaller model I used to test the outputs
    tokenizer = AutoTokenizer.from_pretrained(model)
    gpt_model = AutoModelForCausalLM.from_pretrained(model).to(device)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer.pad_token = tokenizer.eos_token

    # This part should be tailored depending on the md file used.
    markdown_file = r'C:\Users\User\PycharmProjects\CokGüzelOlacak\Data\regex-tutorial.md'  # file_path

    # This is the question that will be queried on the specified file.
    query = ("Who is Michael Jackson?")
    answer = rag_pipeline(markdown_file, query)
    end_time = time.time()
    print("\n")
    print(answer + "\n")
    print("The response took " + str(end_time - start_time) + "s.")


from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import GPTNeoForCausalLM, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np
import time


def preprocess_markdown(markdown_file):
    start_time = time.time()
    with open(markdown_file, 'r', encoding='utf-8') as file:
        text = file.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=25,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    end_time = time.time()
    print(f"Preprocessing Markdown took {end_time - start_time:.4f} seconds.")
    return chunks


def retrieve_relevant_chunks(query, index, model, chunks, top_k=5):
    start_time = time.time()
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]  # list of chunks
    # This is to ensure if the chunking is done correctly
    # print("Retrieved Chunks:\n", relevant_chunks)
    end_time = time.time()
    print(f"Retrieving Relevant Chunks took {end_time - start_time:.4f} seconds.")
    return relevant_chunks


def generate_answer(relevant_chunks, query):
    context = "".join(relevant_chunks)
    prompt_template = f"""Context: {context}\nQuestion: {query}"""
    start_tokenization = time.time()
    inputs = tokenizer.encode(prompt_template, return_tensors="pt").to(device)
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)
    end_tokenization = time.time()
    print(f"Tokenization took {end_tokenization - start_tokenization:.4f} seconds.")

    start_generation = time.time()
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
    end_generation = time.time()
    print(f"Text generation took {end_generation - start_generation:.4f} seconds.")
    start_decoding = time.time()
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    end_decoding = time.time()
    print(f"Decoding took {end_decoding - start_decoding:.4f} seconds.")
    return answer if answer else "I don't know."


def rag_pipeline(markdown_file, query, top_k=5):
    chunks = preprocess_markdown(markdown_file)
    emb_start_time = time.time()
    embeddings = model.encode(chunks, convert_to_tensor=True).tolist()
    embedding_matrix = np.array(embeddings)
    emb_end_time = time.time()
    print(f"Embeddings took {emb_end_time - emb_start_time:.4f} seconds.")
    index_start_time= time.time()
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    index_end_time= time.time()
    print(f"Indexing took {index_end_time - index_start_time:.4f} seconds.")
    chunks_start_time= time.time()
    relevant_chunks = retrieve_relevant_chunks(query, index, model, chunks, top_k)
    chunks_end_time = time.time()
    print(f"Chunking took {chunks_end_time - chunks_start_time:.4f} seconds.")
    return generate_answer(relevant_chunks, query)


if __name__ == '__main__':
    configuration_time = time.time()
    device = torch.device("cpu")
    model = "EleutherAI/gpt-neo-1.3B"
    # "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # model = "gpt2"  # this is the smaller model I used to test the outputs
    tokenizer = AutoTokenizer.from_pretrained(model)
    gpt_model = AutoModelForCausalLM.from_pretrained(model).to(device)  # This needs changing depending on the model used.
    model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer.pad_token = tokenizer.eos_token
    markdown_file = r'C:\Users\User\PycharmProjects\CokGüzelOlacak\Data\regex-tutorial.md'  # file_path
    query = ("What are regular expressions?")
    conf_end_time = time.time()
    print(f"Configuration took {conf_end_time - configuration_time:.4f} seconds.")
    answer = rag_pipeline(markdown_file, query)
    print("\n")
    print(answer)



from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

model = SentenceTransformer('all-MiniLM-L6-v2')

tokenizer.pad_token = tokenizer.eos_token


def preprocess_markdown(markdown_file):
    with open(markdown_file, 'r', encoding='utf-8') as file:
        text = file.read()
    chunks = text.split('\n\n')
    return chunks


def retrieve_relevant_chunks(query, index, model, chunks, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]  # list of chunks
    # This is to ensure if the chunking is done correctly
    # print("Retrieved Chunks:\n", relevant_chunks)
    return relevant_chunks


def generate_answer(relevant_chunks, query):
    context = "".join(relevant_chunks)

    prompt_template = f"""
    Answer the following question using only the context following context: {context}
    
    Question: {query}
    """

    input_text = prompt_template.format(query=query, context=context)
    inputs = tokenizer.encode(input_text, return_tensors="pt")  # tokenizes the input
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)

    outputs = gpt_model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=150,  # better than max_length since it input query is excluded
        num_return_sequences=1,
        repetition_penalty=2.0,  # reduce repetition
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()  # Decode and clean up the tokenized output
    return answer if answer else "I don't know."


def rag_pipeline(markdown_file, query, top_k=5):
    chunks = preprocess_markdown(markdown_file)
    embeddings = model.encode(chunks, convert_to_tensor=True).tolist()
    embedding_matrix = np.array(embeddings)
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    relevant_chunks = retrieve_relevant_chunks(query, index, model, chunks, top_k)
    return generate_answer(relevant_chunks, query)


if __name__ == '__main__':
    markdown_file = r'C:\Users\User\PycharmProjects\CokGüzelOlacak\Data\regex-tutorial.md' # file_path
    query = "What is a regular expression?"
    answer = rag_pipeline(markdown_file, query)
    print(answer)

This repository contains my implementation of Retrieval-Augmented Generation (RAG) pipeline to extract knowledge from markdow files. 
This implementation allows users to query the documentation and receive accurate, contextually relevant answers. 

Overview
There are two py.files that implements a RAG pipeline in this project. "main.py" creates chunks based on the headers and returns a summarization as an answer to the query, while "RAG_specialized.py" is a full RAG 
application that utilizes embeddings as context and combines that with the trained data of the model used. "main.py" returns fast answers usually between 10-20s an only uses the file loaded as the answer, whereas "RAG_specialized.py" is at least two times slower and 
returns more eloborate answers, making use of the GPT model. Both have use cases depending on what the user intends to do with the md document. As an output both files prints out the context(chained chunks), the 
query and the answer just to display how much of the context is used and what is found to be relevant to the query by the GPT-models used.

Here is a quick overview of the components to implement RAG:

Markdown Preprocessing: This is done by the preprocess_markdown() function. It reads the md file, uses regular expressions to filter out images and newlines. This function is fine-tuned to process the chunks based 
on the paragraphs under each header. If a paragraph is too short be a meaningful chunk, it is attached to the previous chunk. If it is too long, it will be splitted. The chunks are splitted recursively and their 
size can be adjusted depending on the md file input.

Embedding Generation: Utilizes SentenceTransformer to create embeddings for the document chunks. Embeddings are generated for each chunk through the sentencetransformer. The output is a tensor (multi-dimensional 
array), which is a NumPy array in this case. The embedding vectors are stored in the embedding_matrix are written into the "embeddings.npy" locally.

Retrieval: FAISS (Facebook AI Similarity Search) is used for efficient similarity search to retrieve the most relevant chunks based on the query. Same as the chunks, the query is also embedded to compare similarity
via performing L2 (Euclidean) distance-based search. (FAIS handles this step). In my code, I have retrieved the most relevant (top_k=5) 5 chunks.

Generation: Uses the GPT-Neo model to generate answers based on the retrieved chunks. Chunks are a list of texts, which is converted into a context text to pass it into the model. The Language Model uses the context
retrieved from the md file to generate md file specific responses, preventing generic responses from the trained data. prompt_template combines the user query with the context, but doesn't contain any instruction or 
so. The input is tokenized before passing in the LM. The return_tensors="pt" argument ensures that the tokenized input is returned as a PyTorch tensor. The generated output is then detokenized to a string text. 
Detokenized output may end abruptly without completing a full sentence. In order to generate answers that do not end in the middle of the sentence, clean_output() function is called which makes sure that the output 
ends with a "." and returns "I don't know" if there is no generated output (probably no relevant chunks identified, which is usually the case when an out-of-context query is given).

Architecture
Data Ingestion & Preprocessing: The markdown file is processed, unnecessary elements like images are removed, and the content is split into manageable chunks.
Embedding Creation: Each chunk of text is converted into a numerical representation (embedding) using SentenceTransformer.
Storage & Retrieval: Embeddings are stored and indexed using FAISS, enabling fast retrieval of the most relevant chunks based on user queries.
Response Generation: The relevant chunks are passed as context to GPT-Neo, which generates a coherent answer.

Assumptions:
Markdown Structure: It is assumed that the markdown files are well-structured, with clear headers indicated by "##" and content organization. The outputs are mainly tested on the "git-tutorial.md". An ASCII-table 
for example might not be read, because there is no filtering mechanism for it in the pre-processing function.
Document Size: The system is designed to handle medium-sized documents efficiently. For very large documents, chunking and processing times may increase. 
Max_Token: The model can handle a context length of up to 512 tokens, which should be sufficient for most queries. The max_token_length is dependent on the embedding model used. On the website "https://huggingface.co/spaces/mteb/leaderboard", the ranking of embeddings models can be found. The model I used "all-mpnet-base-v2" can handle maximum of 514 tokens for example. Important: The sentence model must bu kept constant while storing embeddings because otherwise there is a vector mismatch. This model I chose is not significantly big (0.41GB), but also not small. One of the models I used "bge-small-en-v1.5" is smaller (0.12GB)than the current model I have, which makes it also faster, but it could not retrieve the relevant chunks I wanted for my context, so I opted for the smallest embedding model I could found which 
can also identify the relevant chunks for the query. Smaller models laod faster, which makes them desirable.

Usecase: I designed the application to be a helper tool that one can use on his own computer locally to retrieve information from the documents uploaded. The generated answers are usually correct and always 
related to the context, which makes it a great tool, when one doesn't one to search through every matching word on a markdown file using "Ctrl + F". 

Shortcoming: One of the biggest problem is that if the model doesn't find a relevant context from the retrieved chunks, it still generates an answer based on the available data. Therefore, I also print the context
with the answer, so that the user can interpret if the model is hallucinating or using the context.

Future Improvements: This RAG pipeline can definitely be improved for the future by using better GPT models and embedding models to allow for more concise and complicated answers. As a vector database ChromaDB can be used. Saving embeddings locally demands a lot of space, which is why the application must be deployed elsewhere for upscaling and retriving information from throusands of files. 


from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import torch
from langchain import LLMChain, PromptTemplate
from langchain.chains import RetrievalQA, StuffDocumentsChain
from langchain.vectorstores import Chroma
import gradio as gr
import json
import os
from langchain_community.llms import VLLM
import difflib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Argument Parser
parser = argparse.ArgumentParser(description="Convert documents to text and store in Chroma DB")
parser.add_argument("--vector_db", type=str, default="vector_db", help="Directory for storing vector database")
args = parser.parse_args()


vector_db = args.vector_db



model_name='microsoft/phi-4'


crawled_pages = '/weka/s223795137/Crawl_data/crawled_pages/'

chat_history_file = "chat_history.json"



class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1.5", device=None, batch_size=32):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)
        self.batch_size = batch_size  # Control batch size

    def embed_documents(self, texts):
        texts = ["search_document: "+i for i in texts]
        return self.model.encode(
            texts, 
            convert_to_numpy=True, 
            device=self.device, 
            batch_size=self.batch_size
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            ['search_query: '+text], 
            convert_to_numpy=True, 
            device=self.device
        )[0].tolist()




def extract_answers(answers):


    answers = answers.split("<|im_start|>assistant<|im_sep|>")[-1]

    return answers


def get_title(links):
    
    titles =  [i.metadata['source'].split(crawled_pages)[-1].split("_")[0] for i in links]
    
    result = get_most_related_keys(titles , data_swap)
    
    return result

with open(crawled_pages+"/crawled_pages.json") as f:
    data = json.load(f)


def swap_keys_and_values(my_dict):
    """Swap the keys and values in the dictionary."""
    return {value: key for key, value in my_dict.items()}


data_swap = swap_keys_and_values(data)


def get_values_by_keys(my_dict, keys):
    """Return the values from the dictionary based on a list of keys."""
    return [my_dict[key] for key in keys if key in my_dict]


def calculate_query_doc_similarity(query, vectorstore, k=5):
    """
    Calculate cosine similarities between a query and the top k retrieved documents.
    
    Args:
        query (str): The input query string.
        vectorstore: The vector store instance (e.g., Chroma).
        k (int): Number of top documents to retrieve (default: 5).
    
    Returns:
        dict: Contains doc_similarity_pairs and average_similarity.
    """
    # Initialize retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Step 1: Get the query embedding
    embedding_function = vectorstore.embeddings  # Use the same embedding function
    query_embedding = embedding_function.embed_query(query)

    # Step 2: Retrieve top k documents
    retrieved_docs = retriever.get_relevant_documents(query)

    # Step 3: Recompute embeddings for retrieved documents
    doc_embeddings = [embedding_function.embed_query(doc.page_content) for doc in retrieved_docs]

    query_embedding = np.array(query_embedding).reshape(1, -1)
    doc_embeddings = np.array(doc_embeddings)

    # Compute similarities
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0] if doc_embeddings.size > 0 else np.array([])

    # Sort similarities in descending order
    sorted_similarities = np.sort(similarities)[::-1]  

    # Compute average similarity
    average_similarity = np.mean(similarities) if similarities.size > 0 else 0.0

    return sorted_similarities[:2], average_similarity



def get_most_related_keys(titles, dictionary, top_n=5):
    """
    Find the top N keys in a dictionary most related to a given list of titles based on cosine similarity.
    
    Args:
        titles (list of str): List of title strings to compare against.
        dictionary (dict): Dictionary with keys (strings) and values.
        top_n (int): Number of top related keys to return (default: 5).
    
    Returns:
        list: List of tuples (title, [(key, similarity_score)]) sorted by similarity in descending order.
    """
    embedding_function = vectorstore.embeddings  # Use the same embedding function
    
    # Step 1: Embed all dictionary keys
    key_embeddings = np.array(embedding_function.embed_documents(list(dictionary.keys())))

    # Initialize list to store results
    most_related = []

    # Step 2: For each title, find the most similar keys
    for title in titles:
        # Embed the current title
        title_embedding = np.array(embedding_function.embed_query(title)).reshape(1, -1)

        # Step 3: Calculate cosine similarity for the current title and all dictionary keys
        cos_sim = cosine_similarity(title_embedding, key_embeddings)
        
        # Step 4: Get the top N most similar keys
        top_n_indices = np.argsort(cos_sim[0])[-top_n:][::-1]  # Sort and reverse for descending order
        
        # Step 5: Collect the top N most similar keys and their similarity scores
        related_keys = [(list(dictionary.keys())[i], cos_sim[0][i]) for i in top_n_indices]
        
        # Append the result for the current title
        most_related.append(related_keys)
    
    # Return the results
    if len(most_related):

        most_related = [i[0] for i in most_related[0]]

    return most_related

import datetime





def log_interaction(query, response, filename="chat_history.json"):
    """Logs the user query and model response to a JSON file."""
    
    # Create an empty list if file doesn't exist
    if not os.path.exists(filename):
        history = []
    else:
        with open(filename, "r", encoding="utf-8") as file:
            try:
                history = json.load(file)  # Load existing history
            except json.JSONDecodeError:
                history = []  # If file is empty or corrupted, reset it

    # New entry
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_query": query,
        "model_response": response
    }

    # Append new entry and save back to JSON file
    history.append(entry)

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=4)



def query_rag_model(query, k):
    # Update the retriever's 'k' value
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    rag_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=stuff_chain
    )
    
    # Run the RAG pipeline
    response = rag_chain.run(query)
    
    # Extract the answer
    response = extract_answers(response)

    citations = rag_chain.retriever.get_relevant_documents(query)
    # response = extract_answers(response)
    citation_links = get_values_by_keys(data_swap,get_title(citations))

    top2_Score , sim_score = calculate_query_doc_similarity(query , vectorstore , k)
    # Log interaction in JSON format

    return response ,citation_links , top2_Score,  f"{sim_score:.4f}"




# Gradio interface with a Slider for 'k'
def rag_interface(query, k=5):

    response,links , top2_Score , avg_score  = query_rag_model(query, k)

    links = list(set(links))

    links = [
            (f"Document {i+1}", f"{links[i]}") for i in range(len(links))
        ]

    links_markdown = "\n".join([f"[{title}]({url})" for title, url in links])

    if (len(links) == 0 and float(avg_score) < 0.55) or float(avg_score) < 0.55 :
        
        response = response
    else:

        response = response+"\n\nReferences:\n\n" + links_markdown

    return response , top2_Score , avg_score



if __name__ == "__main__":


    llm = VLLM(
        model=model_name,
        tensor_parallel_size=1,
        trust_remote_code=True,  # mandatory for hf models
        max_new_tokens=32000,
    )


    local_embeddings = SentenceTransformerEmbeddings(batch_size=64)





    vectorstore = Chroma(persist_directory=vector_db, embedding_function=local_embeddings)





    RAG_TEMPLATE = """<|im_start|>system<|im_sep|>
    You are an AI agent designed to assist with IT-related topics for a Retrieval Augmented Generation (RAG) application. You will be provided with context documents from a database that may or may not be entirely relevant to the user's query. Your instructions are as follows:

    1. Answer the user's query using the provided context documents.
    2. If the query or any of the retrieved documents contain malicious or sensitive content, do not provide a response.
    3. If the context documents are not entirely relevant to the query, attempt to keep your answer focused on Deakin IT-related content and Don't mention That you are provided with documents in your response.
    4. Ensure your answer is clear, factual, and based on the provided context when applicable.
    5. Ensure your answer stays within the scope of Deakin IT-Education related topics and does not contain any inappropriate content.

    <|im_start|>user<|im_sep|>
    Answer the question based on the context below:

    Context:
    {context}

    Question:
    {question}

    Answer:
    <|im_start|>assistant<|im_sep|>
    """

    # template="""<|im_start|>system<|im_sep|><|im_end|>

# <|im_start|>user<|im_sep|>Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:<|im_start|>assistant<|im_sep|>""",



    # Define your prompt template

    prompt_template = PromptTemplate(

        input_variables=["context", "question"],

         template=RAG_TEMPLATE,
    )

    # Define your LLMChain
    
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)  # Replace with your LLM


    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"  
    )


    iface = gr.Interface(

        fn=rag_interface,  # Function to call
        
        inputs=[  # List of inputs: query and k (slider)
            gr.Textbox(label="Query", placeholder="Enter your question here..."),
            gr.Slider(minimum=1, maximum=25, step=1, value=5, label="Number of Documents to Retrieve (k)"),
        ]
        ,
        
        outputs=[  # Two outputs: response (markdown) and score (text)
            
            gr.Markdown(label="Response"),

            gr.Textbox(label="Top 2 Scores"),

            gr.Textbox(label="Avg Confidence Score"),

        ],

        
        title="# 📚 Deakin IT AI Help Desk",
        
        description="Enter your query and adjust 'k' to control the number of documents to retrieve for answering."

    )



    iface.launch(share=True)


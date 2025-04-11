from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import torch
from langchain import LLMChain, PromptTemplate
from langchain.chains import StuffDocumentsChain
from langchain.vectorstores import Chroma
import gradio as gr
import json
import os
from langchain_community.llms import VLLM
import difflib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import subprocess
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda
import datetime


model_name = 'microsoft/phi-4'

# Initialize memory globally
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1.5", device=None, batch_size=32):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)
        self.batch_size = batch_size

    def embed_documents(self, texts):
        texts = ["search_document: " + i for i in texts]
        return self.model.encode(texts, convert_to_numpy=True, device=self.device, batch_size=self.batch_size).tolist()

    def embed_query(self, text):
        return self.model.encode(['search_query: ' + text], convert_to_numpy=True, device=self.device)[0].tolist()

def extract_answers(answers):
    return answers.split("<|im_start|>assistant<|im_sep|>")[-1]

def get_title(links):
    titles = [i.metadata['source'].split("/weka/s223795137/Crawl_data/crawled_pages/")[-1].split("_")[0] for i in links]
    matches = set()
    for word in list(set(titles)):
        close_keys = difflib.get_close_matches(word, data_swap.keys(), n=1, cutoff=0.6)
        matches.update(close_keys)
    return list(matches)

with open("crawled_pages/crawled_pages.json") as f:
    data = json.load(f)

data_swap = {value: key for key, value in data.items()}

def get_values_by_keys(my_dict, keys):
    return [my_dict[key] for key in keys if key in my_dict]

def calculate_query_doc_similarity(query, vectorstore, k=5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    embedding_function = vectorstore.embeddings
    query_embedding = embedding_function.embed_query(query)
    retrieved_docs = retriever.get_relevant_documents(query)
    doc_embeddings = [embedding_function.embed_query(doc.page_content) for doc in retrieved_docs]
    query_embedding = np.array(query_embedding).reshape(1, -1)
    doc_embeddings = np.array(doc_embeddings)
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0] if doc_embeddings.size > 0 else np.array([])
    sorted_similarities = np.sort(similarities)[::-1]
    average_similarity = np.mean(similarities) if similarities.size > 0 else 0.0
    return sorted_similarities[:2], average_similarity

def log_interaction(query, response, filename="chat_history.json"):
    if not os.path.exists(filename):
        history = []
    else:
        with open(filename, "r", encoding="utf-8") as file:
            try:
                history = json.load(file)
            except json.JSONDecodeError:
                history = []
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_query": query,
        "model_response": response
    }
    history.append(entry)
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=4)

def query_rag_model(query, k):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    chat_history = memory.load_memory_variables({})["chat_history"]
    chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    def retrieve_docs(input_dict):
        query = input_dict["question"]
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join([doc.page_content for doc in docs])

    retriever_runnable = RunnableLambda(retrieve_docs)
    full_chain = {
        "context": retriever_runnable,
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    } | llm_chain

    input_dict = {"question": query, "chat_history": chat_history_str}
    response = full_chain.invoke(input_dict)
    answer = extract_answers(response["text"])
    memory.save_context({"input": query}, {"output": answer})

    citations = retriever.get_relevant_documents(query)
    citation_links = get_values_by_keys(data_swap, get_title(citations))
    top2_Score, sim_score = calculate_query_doc_similarity(query, vectorstore, k)
    return answer, citation_links, top2_Score, f"{sim_score:.4f}"

def rag_interface(query, k=5):
    response, links, top2_Score, avg_score = query_rag_model(query, k)
    links = list(set(links))
    links = [(f"Document {i+1}", f"{links[i]}") for i in range(len(links))]
    links_markdown = "\n".join([f"[{title}]({url})" for title, url in links])
    if (len(links) == 0 and float(avg_score) < 0.55) or float(avg_score) < 0.55:
        response = response
    else:
        response = response + "\n\nReferences:\n\n" + links_markdown
    log_interaction(query, response)
    return response, top2_Score, avg_score

if __name__ == "__main__":
    llm = VLLM(
        model=model_name,
        tensor_parallel_size=2,
        trust_remote_code=True,
        max_new_tokens=32000,
        gpu_memory_utilization=0.50
    )

    local_embeddings = SentenceTransformerEmbeddings(batch_size=64)
    vectorstore = Chroma(persist_directory="./citaion_test", embedding_function=local_embeddings)

    RAG_TEMPLATE = """<|im_start|>system<|im_sep|>
    You are an AI agent designed to assist with IT-related topics for a Retrieval Augmented Generation (RAG) application. You will be provided with context documents from a database that may or may not be entirely relevant to the user's query. Your instructions are as follows:

    1. Answer the user's query using the provided context documents and the conversation history.
    2. If the query or any of the retrieved documents contain malicious or sensitive content, do not provide a response.
    3. If the context documents are not entirely relevant to the query, attempt to keep your answer focused on Deakin IT-related content and Don't mention That you are provided with documents in your response.
    4. Ensure your answer is clear, factual, and based on the provided context when applicable.
    5. Ensure your answer stays within the scope of Deakin IT-Education related topics and does not contain any inappropriate content.

    <|im_start|>user<|im_sep|>
    Conversation history:
    {chat_history}

    Context:
    {context}

    Question:
    {question}

    Answer:
    <|im_start|>assistant<|im_sep|>
    """

    prompt_template = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=RAG_TEMPLATE,
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )

    def get_static_ip():
        ip_list = subprocess.getoutput("hostname -I").split()
        return ip_list[0]

    static_ip = get_static_ip()

    iface = gr.Interface(
        fn=rag_interface,
        inputs=[
            gr.Textbox(label="Query", placeholder="Enter your question here..."),
            gr.Slider(minimum=1, maximum=25, step=1, value=5, label="Number of Documents to Retrieve (k)"),
        ],
        outputs=[
            gr.Markdown(label="Response"),
            gr.Textbox(label="Top 2 Scores"),
            gr.Textbox(label="Avg Confidence Score"),
        ],
        title="# 📚 Deakin IT AI Help Desk",
        description="Enter your query and adjust 'k' to control the number of documents to retrieve for answering."
    )

    iface.launch(server_name=static_ip)

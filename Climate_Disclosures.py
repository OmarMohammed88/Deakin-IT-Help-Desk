# from langchain.llms import CTransformers
# from langchain.llms import HuggingFacePipeline
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
import torch
from langchain import LLMChain, PromptTemplate
from langchain.chains import RetrievalQA, StuffDocumentsChain
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.vectorstores import Chroma
import gradio as gr
import json
import os
from langchain_community.llms import VLLM




# model_name="/home-old/s223795137/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa",  

model_name='microsoft/phi-4'

# model_name  = 'meta-llama/Llama-3.2-3B-Instruct'

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=32000,device='cuda')




# llm = HuggingFacePipeline(pipeline=pipe)



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




import datetime

chat_history_file = "Climate_Disclosures.json"



def log_interaction(query, response, filename="Climate_Disclosures.json"):
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

    # Log interaction in JSON format
    log_interaction(query, response)

    return response


# def store_rating(query, response, rating):
#     """Stores the rating after the response is shown."""
    
#     log_interaction(query, response, rating)
#     return "Your feedback has been recorded! ✅"


# Gradio interface with a Slider for 'k'
def rag_interface(query, k=5):
    response = query_rag_model(query, k)
    return response



if __name__ == "__main__":


    llm = VLLM(
        model=model_name,
        tensor_parallel_size=1,
        trust_remote_code=True,  # mandatory for hf models
        max_new_tokens=32000,
    )


    local_embeddings = SentenceTransformerEmbeddings(batch_size=64)





    vectorstore = Chroma(persist_directory="./climate_db", embedding_function=local_embeddings)


    # retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Fetch top 5 chunks



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



    # Define your prompt template

    prompt_template = PromptTemplate(

        input_variables=["context", "question"],
        template="""<|im_start|>system<|im_sep|><|im_end|>

    <|im_start|>user<|im_sep|>Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:<|im_start|>assistant<|im_sep|>""",
        # template=RAG_TEMPLATE,
    )

    # Define your LLMChain
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)  # Replace with your LLM


    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"  
    )


    iface = gr.Interface(
    # gr.Markdown("# 📚 Deakin IT AI Help Desk")
    
    fn=rag_interface,   # Function to call
    inputs=[            # List of inputs: query and k (slider)
        gr.Textbox(label="Query", placeholder="Enter your question here..."),
        gr.Slider(minimum=1, maximum=25, step=1, value=25, label="Number of Documents to Retrieve (k)"),


    ], 
    outputs="text",     # Output type (text box)
    title="# 📚 Climate Disclosures",
    description="Enter your query and adjust 'k' to control the number of documents to retrieve for answering."
)


    iface.launch(share=True)


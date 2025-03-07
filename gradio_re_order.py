from langchain.llms import CTransformers
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
import torch
from langchain import LLMChain, PromptTemplate
from langchain.chains import RetrievalQA, StuffDocumentsChain
from langchain.llms import CTransformers
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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

llm = VLLM(
    model=model_name,
    tensor_parallel_size=2,
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=32000,
    gpu_memory_utilization=0.8,
)





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


local_embeddings = SentenceTransformerEmbeddings(batch_size=64)





vectorstore = Chroma(persist_directory="./chroma_db_pages_v3_2.5k", embedding_function=local_embeddings)


# retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Fetch top 5 chunks





# Define your prompt template

prompt_template = PromptTemplate(

    input_variables=["context", "question"],
    template="""<|im_start|>system<|im_sep|><|im_end|>

<|im_start|>user<|im_sep|>Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:<|im_start|>assistant<|im_sep|>""",
 
)

# Define your LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt_template)  # Replace with your LLM



class ReversedStuffDocumentsChain(StuffDocumentsChain):
    def _call(self, inputs: dict) -> dict:
        # Debugging: Check the inputs dictionary structure
        # print(f"Inputs received in _call/: {inputs}")

        # Ensure that the "input_documents" key exists in inputs
        # if "input_documents" not in inputs:
            # raise KeyError("The key 'input_documents' was not found in inputs")

        # Get the documents from the inputs (input_documents)
        documents = inputs["input_documents"]
        
        # Reverse the order of the documents
        reversed_documents = documents[::-1]
        
        # Pass the reversed documents back to the original StuffDocumentsChain logic
        # Convert list of documents into a single string if necessary
        inputs["input_documents"] = reversed_documents
        
        # Pass the adjusted inputs to the parent class
        return super()._call(inputs)
    


reversed_stuff_chain = ReversedStuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context"  # This should match the input variable in your prompt
)


stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context"  
)


def extract_answers(answers):


    answers = answers.split("<|im_start|>assistant<|im_sep|>")[-1]

    return answers




import datetime

chat_history_file = "chat_history_re_order.json"



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




def query_rag_model(query, k , stuff_chain_type):
    # Update the retriever's 'k' value
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    rag_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=stuff_chain_type
    )
    
    # Run the RAG pipeline
    response = rag_chain.run(query)
    
    # Extract the answer
    response = extract_answers(response)

    # Log interaction in JSON format
    log_interaction(query, response,chat_history_file)

    return response


# def store_rating(query, response, rating):
#     """Stores the rating after the response is shown."""
    
#     log_interaction(query, response, rating)
#     return "Your feedback has been recorded! ✅"


# Gradio interface with a Slider for 'k'
def rag_interface(query, k=5):
    response_normal_order = query_rag_model(query, k , stuff_chain)
    response_re_order = query_rag_model(query, k , reversed_stuff_chain)

    return response_normal_order , response_re_order



# Set up the Gradio interface with a Slider for 'k'
# iface = gr.Interface(
#     # gr.Markdown("# 📚 Deakin IT AI Help Desk")
    
#     fn=rag_interface,   # Function to call
#     inputs=[            # List of inputs: query and k (slider)
#         gr.Textbox(label="Query", placeholder="Enter your question here..."),
#         gr.Slider(minimum=1, maximum=25, step=1, value=5, label="Number of Documents to Retrieve (k)"),


#     ], 
#     outputs="text",     # Output type (text box)
#     title="# 📚 Deakin IT AI Help Desk",
#     description="Enter your query and adjust 'k' to control the number of documents to retrieve for answering."
# )



with gr.Blocks() as iface:
    gr.Markdown("# 📚 Deakin IT AI Help Desk")
    gr.Markdown("Enter your query and adjust 'k' to control the number of documents to retrieve for answering.")

    with gr.Row():
        query = gr.Textbox(label="Query", placeholder="Enter your question here...")
        k = gr.Slider(minimum=1, maximum=25, step=1, value=5, label="Number of Documents to Retrieve (k)")

    submit_button = gr.Button("Submit")

    with gr.Row():
        normal_response = gr.Textbox(label="Normal K Retrieved", placeholder="Response with normal document order...")
        reversed_response = gr.Textbox(label="Reversed K Retrieved", placeholder="Response with reversed document order...")

    submit_button.click(rag_interface, inputs=[query, k], outputs=[normal_response, reversed_response])


if __name__ == "__main__":
    iface.launch(share=True)


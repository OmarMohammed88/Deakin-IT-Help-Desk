import argparse
import glob
import os
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import torch
from langchain_chroma import Chroma

# Argument Parser
parser = argparse.ArgumentParser(description="Convert documents to text and store in Chroma DB")
parser.add_argument("--input-directory", type=str, required=True, help="Directory containing documents")
parser.add_argument("--vector_db", type=str, default="vector_db", help="Directory for storing vector database")
args = parser.parse_args()

directory_path = args.input_directory
OUTPUT_DIR =Path(args.vector_db)
VECTOR_DATABASE = Path(args.vector_db)

file_patterns = ['*.txt', '*.html', '*.docx', '*.pdf']
embedding_batch_size = 64

# List to store all the files
all_files = []

# Loop through each pattern and retrieve matching files
for pattern in file_patterns:
    files = glob.glob(os.path.join(directory_path, '**', pattern), recursive=True)
    all_files.extend(files)

input_paths = all_files

# Initialize DocumentConverter for supported formats
doc_converter = DocumentConverter(
    allowed_formats=[
        InputFormat.PDF,
        InputFormat.IMAGE,
        InputFormat.DOCX,
        InputFormat.HTML,
        InputFormat.PPTX,
        InputFormat.ASCIIDOC,
        InputFormat.CSV,
        InputFormat.MD,
    ],
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=StandardPdfPipeline,
        ),
        InputFormat.DOCX: WordFormatOption(
            pipeline_cls=SimplePipeline
        ),
    },
)

def process_text_file(file_path: str) -> dict:
    """Process a .txt file and return a simple result structure."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {
            "status": "success",
            "filename": Path(file_path).name,
            "document": content
        }
    except Exception as e:
        return {
            "status": "failed",
            "filename": Path(file_path).name,
            "error": str(e)
        }

txt_files = [f for f in input_paths if f.lower().endswith('.txt')]
non_txt_files = [f for f in input_paths if not f.lower().endswith('.txt')]

conv_results = []
if non_txt_files:
    conv_results.extend(doc_converter.convert_all(non_txt_files))

for txt_file in txt_files:
    txt_result = process_text_file(txt_file)
    conv_results.append(txt_result)

def save_converted_data(docs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        if isinstance(doc, dict):
            with (output_dir / f"{doc['filename']}.md").open("w") as fp:
                fp.write(doc['document'])
        else:
            with (output_dir / f"{doc.input.file.stem}.md").open("w",encoding="utf-8") as fp:
                fp.write(doc.document.export_to_markdown())

save_converted_data(conv_results, OUTPUT_DIR)

loader = DirectoryLoader(OUTPUT_DIR, glob="*.md")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=0)
all_splits = text_splitter.split_documents(documents)

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1.5", device=None, batch_size=32):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)
        self.batch_size = batch_size

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
VECTOR_DATABASE.mkdir(parents=True, exist_ok=True)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings, persist_directory=str(VECTOR_DATABASE))

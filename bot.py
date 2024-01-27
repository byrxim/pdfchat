from langchain_community.llms import HuggingFaceHub
import textwrap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
import os
import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

document = PdfReader("48lawsofpower.pdf")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.8,"max_length":512})

print(llm.invoke("What is Flan"))

raw_text=''

for i, page in enumerate(document.pages):
    text=page.extract_text()
    if text:
        raw_text += text

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
docs = text_splitter.split_text(raw_text)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_texts(docs, embeddings)
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain


query="how to win"

docRes=db.similarity_search(query, k=1)

chain = load_summarize_chain(llm, chain_type="map_reduce")
chain(docRes)

chain2 = load_qa_chain(llm, chain_type="refine", verbose=True)

chain2({'input_documents': docRes, 'question': query})
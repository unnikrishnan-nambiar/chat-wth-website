import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document  # Import Document class

# Load environment variables from .env file (Optional)
load_dotenv()

# Optional
# OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# Function to read text from multiple PDF files
def read_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def main():
    # Set the title and subtitle of the app
    st.title('🦜🔗 Chat With PDF Files')
    st.subheader('Upload your PDF files, ask questions, and receive answers directly from the PDFs.')

    user_question = st.text_input("Ask a question (query/prompt)")

    # File uploader for multiple PDFs
    pdf_files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

    if pdf_files and user_question:
        if st.button("Submit Query", type="primary"):
            ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
            DB_DIR: str = os.path.join(ABS_PATH, "db")

            # Read and combine text from the PDFs
            raw_text = read_pdfs(pdf_files)

            # Split the loaded text into chunks
            text_splitter = CharacterTextSplitter(separator='\n', chunk_size=2000, chunk_overlap=40)
            text_chunks = text_splitter.split_text(raw_text)

            # Wrap the text chunks into Document objects (this is needed for Chroma)
            documents = [Document(page_content=chunk) for chunk in text_chunks]

            # Create Ollama embeddings
            ollama_embeddings = OllamaEmbeddings(model="mistral")

            # Create a Chroma vector database from the documents
            vectordb = Chroma.from_documents(documents=documents, 
                                            embedding=ollama_embeddings,
                                            persist_directory=DB_DIR)

            vectordb.persist()

            # Create a retriever from the Chroma vector database
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})

            # Use a mistral llm from Ollama
            llm = Ollama(model="mistral")

            # Create a RetrievalQA from the model and retriever
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

            # Run the prompt and return the response
            response = qa(user_question)
            st.write(response)

if __name__ == '__main__':
    main()
    
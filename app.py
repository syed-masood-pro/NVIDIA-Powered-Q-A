import streamlit as st
import os
import time
import tempfile
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

#Loading environment variables from .env file
load_dotenv()

#Setting the title, icon, and layout for the web app
st.set_page_config(page_title="NVIDIA NIM RAG", page_icon="🤖", layout="wide")

st.title("NVIDIA-Powered Q&A: Chat with Your Documents🤖")

#Applying custom CSS for styling the app's appearance
st.markdown("""
<style>
    .stApp { background: #04062F; }
    .stButton>button {
        background-color: #76b900; color: white; border-radius: 12px;
        padding: 10px 24px; border: none; font-weight: bold;
    }
    .stTextInput>div>div>input { border-radius: 12px; }
    .st-emotion-cache-1629p8f, .st-emotion-cache-13ln4jf { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


def vector_embedding(uploaded_files):
    #Checking if files were uploaded
    if not uploaded_files:
        st.sidebar.error("Please upload one or more PDF files.")
        return

    #Checking for the NVIDIA API key in environment variables
    if not os.getenv("NVIDIA_API_KEY"):
        st.error("NVIDIA_API_KEY not found. Please set it in your .env file.")
        st.stop()

    #Showing a spinner while processing the documents
    with st.spinner("Processing Documents and Creating Embeddings..."):
        try:
            #Initializing the NVIDIA embeddings model
            st.session_state.embeddings = NVIDIAEmbeddings()
            
            all_docs = []
            #Using a temporary directory for safe file handling
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    #Using a temporary directory to safely handle file uploads
                    temp_filepath = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_filepath, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                     #Loading the PDF using PyPDFLoader
                    loader = PyPDFLoader(temp_filepath)
                    docs = loader.load()
                    all_docs.extend(docs)

            #Checking if any documents were successfully loaded
            if not all_docs:
                st.warning("Could not extract text from the uploaded PDFs.")
                return

            #Spliting documents into chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(all_docs)
            
            #Creating FAISS vector store
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.sidebar.success("Vector Store DB is ready!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            #Cleaning up session state on error to prevent partial states
            if "vectors" in st.session_state:
                del st.session_state["vectors"] #Clearing partial state on error


#Creating sidebar for File Upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload your PDF documents",
        type="pdf",
        accept_multiple_files=True
    )
    #Button to trigger the embedding process
    if st.button("Process Documents"):
        vector_embedding(uploaded_files)


try:
    #Initializing the NVIDIA chat model
    llm = ChatNVIDIA(model='meta/llama-3.1-70b-instruct',api_key=os.getenv("NVIDIA_API_KEY"))

    #Defining the prompt template for the RAG chain
    prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the question based on the provided context only.
        Please provide the most accurate and concise response based on the question. If the information is not in the context, say "I don't have enough information to answer that."
        <context>
        {context}
        </context>
        Question: {input}
        """
    )
except Exception as e:
    st.error(f"Failed to initialize the LLM. Please check your NVIDIA API key. Error: {e}")
    st.stop()


if "vectors" in st.session_state:
    st.header("Ask a Question")
    prompt_input = st.text_input("Enter your question about the documents:")

    if prompt_input:
        with st.spinner("Searching for the answer..."):
            try:
                #Creating the document chain that combines the prompt and LLM
                document_chain = create_stuff_documents_chain(llm, prompt_template)
                #Creating a retriever from the FAISS vector store
                retriever = st.session_state.vectors.as_retriever()
                #Creating the final retrieval chain that combines the retriever and document chain
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                start_time = time.process_time()
                #Invoking the chain with the user's question
                response = retrieval_chain.invoke({'input': prompt_input})
                elapsed_time = time.process_time() - start_time
                
                st.write(f"Response generated in: {elapsed_time:.2f} seconds")
                st.success(response['answer'])

                #Creating an expander to show the source documents used for the answer
                with st.expander("Show Document Context"):
                    for i, doc in enumerate(response['context']):
                        st.markdown(f"**Source Document {i+1}:**")
                        st.write(doc.page_content)
                        st.write("---")
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    #Displaying initial message to guide the user
    st.info("Please upload your PDF documents and click 'Process Documents' in the sidebar to get started.")


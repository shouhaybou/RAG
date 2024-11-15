# Import necessary libraries
import openai
import streamlit as st
import fitz  # PyMuPDF for PDF extraction
from dotenv import load_dotenv
import os
from groq import Groq
from microagent import Microagent, Agent

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client to get available models to choose from
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
models = groq_client.models.list()
ids = [item.id for item in models.data]

default_model = "llama-3.2-90b-text-preview"

# Initialize Microagent client
client = Microagent(llm_type='groq')

# Initialize session state for article
if 'article' not in st.session_state:
    st.session_state.article = ""
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = default_model

# Callback function to update model selection
def update_model():
    st.session_state.selected_model = st.session_state.model_dropdown

# Function to extract text from a limited number of pages in a PDF file
def pdf_search(file, max_pages=10):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for i, page in enumerate(doc):
        if i >= max_pages:  # Limit number of pages processed
            break
        text += page.get_text()
    doc.close()
    return text

# Function to split large text into smaller chunks based on token size
def chunk_text(text, max_tokens=5000):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        # Approximate token count by word count; adjust max_tokens if needed
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Define Agent for PDF extraction and analysis
pdf_agent = Agent(
    name="PDF Assistant",
    instructions="""Your role is to read, extract, and analyze information from the provided PDF content.
                    Organize the content by main themes and key points include details for each main themes.""",
    functions=[pdf_search],
    model=st.session_state.selected_model
)

# Define PDF Writer Agent to summarize extracted content
pdf_writer_agent = Agent(
    name="PDF Writer Assistant",
    instructions="""Transform the extracted PDF content into a clear, organized, and production-ready article or summary.
                    Structure content with headlines, subheadings, and ensure readability.""",
    model=st.session_state.selected_model
)

# Updated workflow to process PDF in smaller chunks
def run_workflow(pdf_file):
    print("Running PDF extraction workflow...")
    
    # Step 1: Extract content from the PDF
    pdf_content = pdf_search(pdf_file)
    
    # Step 2: Split the content into smaller chunks if necessary
    text_chunks = chunk_text(pdf_content, max_tokens=5000)
    
    # Step 3: Process each chunk with pdf_agent and pdf_writer_agent
    results = []
    for chunk in text_chunks:
        pdf_response = client.run(agent=pdf_agent, messages=[{"role": "user", "content": chunk}])
        raw_information = pdf_response.messages[-1]["content"]
        
        # Summarize and format the chunk
        publication_response = client.run(agent=pdf_writer_agent, messages=[{"role": "user", "content": raw_information}])
        results.append(publication_response.messages[-1]["content"])

    # Combine results from all chunks
    final_result = "\n\n".join(results)
    st.session_state.article = final_result

# Streamlit app setup
st.set_page_config(page_title="PDF Research Assistant 🔎", page_icon="🔎")
st.title("PDF Research Assistant 🔎")
st.write("Upload a PDF to generate a summarized article with multi-agent RAG.")

# PDF file uploader
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

# Model selection dropdown
selected_model = st.selectbox(
    "Choose a model:",
    options=ids,
    index=ids.index(st.session_state.selected_model),
    key="model_dropdown",
    on_change=update_model
)

# Generate article only when button is clicked and PDF file is uploaded
if st.button("Generate Article") and pdf_file:
    with st.spinner("Generating article..."):
        # Run workflow and get the final generated content
        run_workflow(pdf_file=pdf_file)

# Display the article if it exists in the session state
if st.session_state.article:
    st.markdown(st.session_state.article, unsafe_allow_html=True)

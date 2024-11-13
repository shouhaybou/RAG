# import necessary libraries
import openai
import streamlit as st
# from swarm import Swarm, Agent
from duckduckgo_search import DDGS
from datetime import datetime
# from dotenv import load_dotenv
import os
from groq import Groq
from microagent import Microagent, Agent

load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")
#openai.api_key = st.secrets["GROQ_API_KEY"]



# Initialize Groq client to get available models to choose from
groq_client = Groq(
     api_key=os.getenv("GROQ_API_KEY"),
 )
#groq_client = Groq(
#    api_key=st.secrets["GROQ_API_KEY"],
#)

models = groq_client.models.list()
ids = [item.id for item in  models.data]

default_model = "llama-3.2-90b-text-preview"

# initialize Microagent client
client = Microagent(llm_type='groq')

# initialize search client
ddgs = DDGS()

# Initialize session state for query and article
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'article' not in st.session_state:
    st.session_state.article = ""
if 'selected_model' not in st.session_state:
    # Set the default model if not in session state
    st.session_state.selected_model = default_model


# Search the web for the given query
def search_web(query):
    # print(f"Searching the web for {query}...")
    print("Searching the Web")
    # Duckduckgo Search
    current_date = datetime.now().strftime("%Y-%m")
    results = ddgs.text(f"{query} {current_date}", safesearch = "off", region="wt-wt", max_results = 15)

    if results:
        web_results = ""
        for result in results:
            web_results += f"Title: {result['title']}\nURL: {result['href']}\nDescription: {result['body']}\n\n"
        # print(web_results)
        return web_results.strip()
    else:
        return f"Could not find web results for {query}"

# search the news for the given query
def search_news(query):
    # print(f"Searching the news for {query}...")
    print("Searching the News")
    # Duckduckgo Search
    results = ddgs.news(f"{query}", safesearch = "off", region="wt-wt", max_results = 15)

    if results:
        news_results = ""
        for result in results:
            news_results += f"Title: {result['title']}\nDate: {result['date']}\nURL: {result['url']}\nDescription: {result['body']}\nSource: {result['source']}\n\n"
        # print(web_results)
        return news_results.strip()
    else:
        return f"Could not find news results for {query}"
    
### write a function for extract pdf file

def transfer_to_researcher():
    """Transfer raw information immediately. """
    return researcher_agent

# Define Agent 1 Role: Web search agent to fetch latest news
"""
web_search_agent = Agent(
    name = "Web Search Assistant",
    instructions = Your role is to gather the latest, high-quality articles on the specified topics using the internet search tools provided to you. 
                    Make sure to leverage new insights with unique results, and focus on sources that are authoritative, recent, and relevant.
                    Prioritize searching the news if the user is looking for recent information.,
""" 
# pdf_search_agent
def pdf_search(file):
    text = ""
    doc = PyMuPDF.open(file)
    for page in doc:
        text += page.get_text()
    doc.close()
    return text
    

# web_search_agent.functions.extend(transfer_to_researcher)

# Define Agent 2 Role: Senior Research Analyst
# Define Agent: PDF Content Extractor and Analyzer
pdf_agent = Agent(
    name="PDF Assistant",
    instructions="""Your role is to read, extract, and analyze information from the provided PDF content. 
                    Prioritize the following:
                    1. Extract relevant information that aligns with the user’s query or topic of interest.
                    2. Organize the content into primary themes and key topics.
                    3. Remove duplicate information and flag any contradictory statements.
                    4. Focus on extracting important dates, statistics, and quotes, ensuring accuracy in the context.
                    5. Make sure to provide a structured overview with clear sections for easy readability.""",
    functions=[pdf_search],
    model = st.session_state.selected_model,
    tool_choice = "auto",

)


# Define Agent 3 Role: Editor Agent
pdf_writer_agent = Agent(
    name="PDF Writer Assistant",
    instructions="""Your role is to transform the extracted PDF content into a clear, organized, and production-ready article or summary.
                    You should:
                    1. Structure the content into logical sections with headlines and subheadings based on main themes and topics in the PDF.
                    2. Emphasize clarity and narrative flow, making sure content is easy to read and follows a logical order.
                    3. Provide summaries of key insights at the beginning of each section, and a brief conclusion summarizing the overall content.
                    4. Maintain factual accuracy and coherence, especially when working with fragmented or unstructured data.
                    5. Include important details from the PDF, such as dates, statistics, quotes, and any cited sources.
                    6. Ensure proper formatting, making the text visually accessible and organized for readers to easily follow.""",
    model=st.session_state.selected_model
)
# define agent workflow
# Workflow to process the query or PDF file
def run_workflow(query, pdf_file=None):
    print("Running PDF and Web Research Assistant workflow...")

    # Use pdf_agent if a PDF file is provided
    if pdf_file is not None:
        pdf_content = pdf_search(pdf_file)
        pdf_response = client.run(
            agent=pdf_agent,
            messages=[{"role": "user", "content": pdf_content}],
        )
        raw_information = pdf_response.messages[-1]["content"]
    else:
        # Perform a web search if no PDF is provided
        raw_response = client.run(
            agent=web_search_agent,
            max_turns=3,
            messages=[{"role": "user", "content": f"Search for {query}"}],
        )
        raw_information = raw_response.messages[-1]["content"]

    # Analyze and synthesize the extracted information
    research_analysis_response = client.run(
        agent=researcher_agent,
        messages=[{"role": "user", "content": raw_information}],
    )
    deduplicated_information = research_analysis_response.messages[-1]["content"]

    # Edit and format the synthesized results
    publication_response = client.run(
        agent=writer_agent,
        messages=[{"role": "user", "content": deduplicated_information}],
    )
    return publication_response.messages[-1]["content"]



# Define callback function for model selection
# Streamlit app setup
st.set_page_config(page_title="Web and PDF Research Assistant 🔎", page_icon="🔎")
st.title("Web and PDF Research Assistant 🔎")
st.write("Search for a topic or upload a PDF to generate an article with multi-agent RAG. Note: some models may not support tools.")

# Create two columns for input (query or PDF) and the clear button
col1, col2 = st.columns([3, 1])

# Input section for search query or PDF upload
with col1:
    # Search query input
    query = st.text_input(label="Enter your search query", label_visibility="collapsed", placeholder="Enter your search query (optional if uploading PDF)")
    
    # PDF file uploader
    pdf_file = st.file_uploader("Or upload a PDF file", type="pdf")

    # Model selection dropdown
    selected_model = st.selectbox(
        f"Choose a model. Default: **{default_model}**:",
        options=ids,
        index=ids.index(st.session_state.selected_model),
        key="model_dropdown",
        on_change=update_model
    )
    st.write(f"You have selected the model: **{selected_model}**")

# Clear button
with col2:
    if st.button("Clear"):
        st.session_state.query = ""
        st.session_state.article = ""
        st.session_state.selected_model = default_model
        st.rerun()

# Generate article only when button is clicked
if st.button("Generate Article"):
    with st.spinner("Generating article..."):
        # Check if a PDF file is uploaded; if so, prioritize PDF processing
        if pdf_file:
            generated_content = run_workflow(query="", pdf_file=pdf_file)
            st.session_state.query = "PDF content processed"
        elif query:
            generated_content = run_workflow(query=query)
            st.session_state.query = query
        else:
            st.warning("Please enter a search query or upload a PDF file.")
            generated_content = None
        
        # Store the generated content in the session state if available
        if generated_content:
            st.session_state.article = generated_content

# Display the article if it exists in the session state
if st.session_state.article:
    st.markdown(st.session_state.article, unsafe_allow_html=True)
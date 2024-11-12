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

# load_dotenv()
# openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_key = st.secrets["GROQ_API_KEY"]



# Initialize Groq client to get available models to choose from
# groq_client = Groq(
#     api_key=os.getenv("GROQ_API_KEY"),
# )
groq_client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

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

def transfer_to_researcher():
    """Transfer raw information immediately. """
    return researcher_agent

# Define Agent 1 Role: Web search agent to fetch latest news
web_search_agent = Agent(
    name = "Web Search Assistant",
    instructions = """Your role is to gather the latest, high-quality articles on the specified topics using the internet search tools provided to you. 
                    Make sure to leverage new insights with unique results, and focus on sources that are authoritative, recent, and relevant.
                    Prioritize searching the news if the user is looking for recent information.""",
    functions=[search_web, search_news],
    model = st.session_state.selected_model,
    tool_choice = "auto",
)

# web_search_agent.functions.extend(transfer_to_researcher)

# Define Agent 2 Role: Senior Research Analyst
researcher_agent = Agent(
    name = "Researcher Assistant",
    instructions = """Your role is to analyze and synthesize the raw search results. Prioritize the following:
                    1. Relevance and recency of information.
                    2. Key topics organized by primary themes.
                    3. Deduplication of content while flagging any contradictory statements.
                    4. Enhanced focus on high-authority sources and verifying data across multiple sources.
                    5. Extract important dates, statistics, quotes, and ensure accurate source attribution (including links where appropriate). """,
    model = st.session_state.selected_model
)

# Define Agent 3 Role: Editor Agent
writer_agent = Agent(
    name = "Writer Assistant",
    instructions = """Your role is to transform the de-duplicated research results into a clear, engaging, and production-ready article. 
                    You should:
                    1. Structure content into logical sections with headlines and subheadings.
                    2. Emphasize clarity and narrative flow, using transitions that make sense for readers.
                    3. Summarize key insights briefly at the beginning and end of sections.
                    4. Maintain factual accuracy while making complex topics accessible.
                    5. Include all essential information from the source material, such as dates, quotes, sources, and source links.
                    6. Make sure all text is formatted correctly and rendered properly.""",
    model = st.session_state.selected_model
)

# define agent workflow
def run_workflow(query):
    print("Running Duckduckgo Research Assistant workflow...")

    # search the web
    raw_response = client.run(
        agent = web_search_agent,
        max_turns = 3,
        messages = [{"role": "user", "content": f"Search Duckduckgo for {query}"}],
    )
    # print(news_response)
    raw_information = raw_response.messages[-1]["content"]

    # analyze and synthesize the research results
    research_analysis_response = client.run(
        agent = researcher_agent,
        messages = [{"role": "user", "content": raw_information}],
    )

    # print(research_analysis_response)
    deduplicated_information = research_analysis_response.messages[-1]["content"]

    # Edit and publish the analyzed results
    publication_response = client.run(
        agent = writer_agent,
        messages = [{"role": "user", "content": deduplicated_information}],
    )

    # print(publication_response)
    return publication_response.messages[-1]["content"]


# Define callback function for model selection
def update_model():
    st.session_state.selected_model = st.session_state.model_dropdown

# streamlit app
st.set_page_config(page_title="Duckduckgo Research Assistant 🔎", page_icon="🔎")
st.title("Duckduckgo Web and News Research Assistant 🔎")
st.write("Search for a topic and generate an article with multi-agent RAG. Note: some models may not support tools.")


# Create two columns for the input and clear button
col1, col2 = st.columns([3, 1])

# Search query input
with col1:
    query = st.text_input(label = "Enter your search query",label_visibility = "collapsed",  placeholder = "Enter your search query")

    selected_model = st.selectbox(
        f"Choose a model. Default: **{default_model}**:",
        options = ids,
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
if st.button("Generate Article") and query:
    with st.spinner("Generating article..."):
        # Run workflow and get the final generated content
        generated_content = run_workflow(query)
        # print(generated_content)
        st.session_state.query = query
        st.session_state.article = generated_content

# Display the article if it exists in the session state
if st.session_state.article:
    st.markdown(st.session_state.article, unsafe_allow_html=True)




import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

tools = [arxiv_tool, wiki_tool]

# Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize LLM
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
llm_with_tools = llm.bind_tools(tools=tools)

# Define chatbot function
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

# Streamlit UI
st.title("LangGraph Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    events = graph.stream(
        {"messages": [("user", prompt)]}, stream_mode="values"
    )
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for event in events:
            full_response += event["messages"][-1].content
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""

import streamlit as st
import pandas as pd
from llm import (
    run_rag_query,
    run_graphrag_query,
    run_natural_language_query,
    determine_question_type,
    get_vulnerability_list,
    QUERY_TYPES,
)
from graph import setup_graph

# ------------------------------------------------------------------------
# Functions
# This section contains the functions needed to be called by our streamlit app


def write_messages():
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            create_display(message["content"])


def create_display(response):
    if isinstance(response, dict) or isinstance(response, list):
        if isinstance(response, dict) and "subgraph" in response:
            setup_graph(response["subgraph"])
        else:
            st.dataframe(response, use_container_width=True)

    else:
        st.write(response)


def get_query_type_text(type: QUERY_TYPES):
    match type:
        case QUERY_TYPES.KnowledgeGraph:
            return "Graph Query"
        case QUERY_TYPES.RAG:
            return "Similarity Query"
        case QUERY_TYPES.GraphRAG:
            return "Graph + Similarity Query"


def run_query(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with tab1:
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner(f"Thinking..."):
            type = determine_question_type(prompt)

        with st.spinner(f"Executing using {get_query_type_text(type)}..."):
            with st.chat_message("assistant"):
                match type:
                    case QUERY_TYPES.KnowledgeGraph:
                        response = run_natural_language_query(prompt)
                    case QUERY_TYPES.RAG:
                        response = run_rag_query(prompt)
                    case QUERY_TYPES.GraphRAG:
                        response = run_graphrag_query(prompt)
                    case _:
                        response = (
                            "I am not sure how to execute that query, please try again."
                        )
                create_display(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response, "type": "table"}
                )


# ------------------------------------------------------------------------
# Streamlit
# This section represents our Streamlit App UI and Actions

# Page title
st.set_page_config(
    page_title="SBOM Graph RAG Demo",
    layout="wide",
)

st.title("SBOM Vulnerability Graph RAG Demo")

# # Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "How may I assist you today?",
            "context": "assistant",
        }
    ]

# Setup columns for the two chatbots
tab1, tab2 = st.tabs(["Chat", "Schema"])
with tab1:
    # Setup the chat input
    write_messages()

with tab2:
    st.subheader("SBOM Graph schema")
    st.image("schema.png", use_column_width=True)


# React to user input
if prompt := st.chat_input():
    run_query(prompt)

# Configure the sidebar with the example questions
with st.sidebar:
    st.header("Example Queries")

    kg_option = st.selectbox(
        "Select a Knowledge Graph Query to run:",
        (
            "How many Vulnerabilities exist?",
            "How many 'high' or 'critical' Vulnerabilities are there?",
        ),
    )

    if st.button("Run", key="kg_queries"):
        run_query(kg_option)

    sim_option = st.selectbox("Select a Vulnerability:", get_vulnerability_list())

    if st.button("Find the most similar", key="sim_queries"):
        run_query(f"What Vulnerabilities are similar to '{sim_option}'?")
    if st.button("Show me how they are connected", key="whole_dataset"):
        run_query(
            f"What Vulnerabilities are similar to '{sim_option}' and show me how they are connected"
        )

"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""

import streamlit as st
from llm import (
    run_natural_language_query,
)
from utils import write_messages, create_display

# ------------------------------------------------------------------------
# Functions
# This section contains the functions needed to be called by our streamlit app

# # Store LLM generated responses
if "messages_nlq" not in st.session_state.keys():
    st.session_state.messages_nlq = [
        {
            "role": "assistant",
            "content": "How may I assist you today?",
            "context": "assistant",
        }
    ]

messages = st.session_state.messages_nlq


def run_query(prompt):
    messages.append({"role": "user", "content": prompt})

    with tab1:
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner(f"Executing using natural language query translation ..."):
            with st.chat_message("assistant"):
                response = run_natural_language_query(prompt)
                create_display(response)
                messages.append(
                    {"role": "assistant", "content": response, "type": "table"}
                )


# Page title
st.set_page_config(
    page_title="SBOM Graph RAG Demo",
    layout="wide",
)

st.title("Natural Language Query")
st.write(
    """Using Amazon Bedrock Foundation models, your natural language question will be converted into an openCypher query, which will then be run, and results returned."""
)

tab1, tab2 = st.tabs(["Chat", " "])
with st.container():
    # Setup the chat input
    write_messages(messages)

# React to user input
if prompt := st.chat_input():
    run_query(prompt)

# Configure the sidebar with the example questions
with st.sidebar:
    st.header("Example Queries")

    kg_option = st.selectbox(
        "Select the query to run or enter your own below:",
        (
            "How many Vulnerabilities exist?",
            "How many 'high' or 'critical' Vulnerabilities are there?",
        ),
    )

    if st.button("Run", key="kg_queries"):
        run_query(kg_option)

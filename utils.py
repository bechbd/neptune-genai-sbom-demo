import streamlit as st
from graph import setup_graph


def write_messages(message_state):
    # Display chat messages from history on app rerun
    for message in message_state:
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

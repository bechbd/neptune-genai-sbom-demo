import streamlit as st
from llm import get_vulnerability_list

st.set_page_config(
    page_title="Neptune Generative AI Demo",
    page_icon="👋",
    layout="wide",
)

st.write("# Software Bill Of Materials Generative AI Demo using Amazon Neptune")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)
st.subheader("SBOM Graph schema")
st.image("schema.png", use_column_width=True)
st.session_state.vulnerabiity_list = get_vulnerability_list()

import streamlit as st
from smolagents import CodeAgent, HfApiModel

st.title("Agentic Code Analyst 🤖")

# Use a secret for your token (Set this in HF Space Settings)
import os
token = os.getenv("HF_TOKEN")

if "agent" not in st.session_state:
    # Initialize the brain (Qwen 2.5 Coder is excellent for this)
    model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", token=token)
    st.session_state.agent = CodeAgent(tools=[], model=model)

query = st.text_input("Ask about your code:")
if st.button("Run"):
    with st.spinner("Analyzing..."):
        response = st.session_state.agent.run(query)
        st.write(response)
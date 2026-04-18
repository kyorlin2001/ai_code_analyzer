---
title: Code Analysis Tool
emoji: 🚀
colorFrom: red
colorTo: red
sdk: streamlit
app_file: src/app.py
pinned: false
short_description: Agentic RAG tool for codebase analysis.
license: apache-2.0
---

# 🤖 Agentic Code Analysis Tool

This repository contains an AI-powered agent designed to analyze software projects. 

## 🚀 Live Demo
**[Check out the live app on Hugging Face Spaces](https://huggingface.co/spaces/kyorlin2001/code-analysis-tool)**

## 🛠️ Features
* **Zip Analysis:** Upload a project and let the agent map the architecture.
* **Agentic Reasoning:** Powered by `smolagents` and Qwen-2.5-Coder.
* **Modern Stack:** Built with Python 3.11+ and Streamlit.

## 📥 Local Setup
1. Clone the repo.
2. Create a `.venv` and install `requirements.txt`.
3. Set your `HF_TOKEN` in a `.env` file.
4. Run `streamlit run src/app.py`.
## DocInsights

talk to documents using local LLM with the power of RAG

# Installation
````
pip install -r requirements.txt
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
````
--in a seperate window--
````
ollama pull qwen:0.5b
streamlit run streamlit_app.py
````

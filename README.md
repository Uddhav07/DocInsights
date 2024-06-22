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
ollama pull qwen2:0.5b
ollama pull gemma:2b
python -m streamlit run streamlit_app.py
````

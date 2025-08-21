# 🔐 LLM Security with PyRIT & Ollama  

This repository contains **custom scripts and workflows** built on top of [Microsoft’s PyRIT (Python Risk Identification Tool)](https://github.com/microsoft/pyrit).  
The project is designed to test and evaluate **LLM vulnerabilities** by integrating PyRIT with **Ollama**, enabling automated red-teaming and analysis of model outputs.  

⚠️ **Note:** This repo does **not** include the PyRIT source code. Please follow the setup instructions below to download and configure **PyRIT** and **Ollama** before using the scripts provided here.  

---

## 🚀 Features  
- ✅ Extend PyRIT with new workflows and prompts.  
- ✅ Integration with **Ollama** for local LLM evaluation.  
- ✅ JSON-based result storage for reproducible experiments.  
- ✅ Configurable judge models for assessing harmful/vulnerable content.  

---

## 🛠 Requirements  
- Python 3.9+  
- Git  
- [Ollama](https://ollama.com/download) (installed locally)  
- [PyRIT](https://github.com/microsoft/pyrit) (cloned separately)  

---

## 📦 Installation  

### 1. Clone PyRIT  
First, clone Microsoft’s PyRIT and install dependencies:  
```bash
git clone https://github.com/microsoft/pyrit.git
cd pyrit
pip install -r requirements.txt

Download and install Ollama from the official site:
👉 https://ollama.com/download

Verify installation:

ollama --version

3. Clone This Repository

In a separate folder (not inside PyRIT):

git clone https://github.com/<your-username>/LMM_Security.git
cd LMM_Security

4. Configure Paths

Ensure both pyrit/ and this LMM_Security/ repo are in the same workspace.

Update any import paths in your custom scripts if needed (e.g., point to your local pyrit folder).

▶️ Usage

Start Ollama with your chosen model (example with Mistral):

ollama run mistral


Run the workflow with your custom scripts:

python run_workflow.py --config configs/ollama_config.json


Results will be stored in the results/ directory as JSON logs.

📊 Example Workflow

Send multiple jailbreak prompts to Ollama models (e.g., Mistral, LLaMA 3).

Collect responses in structured JSON format.

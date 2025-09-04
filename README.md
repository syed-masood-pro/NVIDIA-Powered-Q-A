# NVIDIA-Powered-Q-A: Chat with Your Documents

This project is a powerful, document-centric Q&A application built with Streamlit and powered by **NVIDIA NIM** inference microservices. It implements a Retrieval-Augmented Generation (RAG) pipeline that allows you to upload your own PDF documents, ask questions about them, and receive accurate, context-aware answers directly from the content.

![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit)
![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20LangChain%20%7C%20Gemma_2-blueviolet)
![Language](https://img.shields.io/badge/Language-Python_3.11-yellow?logo=python)  

## 📝 Table of Contents
- [📌 About the Project](#-about-the-project)
- [✨ Key Features](#-key-features)
- [🛠️ Requirements](#️-requirements)
- [🚀 How to Run](#-how-to-run)
- [🖼️ Sample Interface](#-sample-interface)
- [📄 License](#-license)
- [📧 Contact](#-contact)
---
## 📌 About the Project

This application demonstrates a complete RAG workflow using cutting-edge tools. It leverages LangChain to orchestrate the pipeline, NVIDIA NIM for both generating embeddings and for the chat model (Llama 3.1 70B), and a FAISS vector store for efficient similarity searches. When a user uploads PDFs, the application processes and chunks the text, creates vector embeddings, and stores them. When a question is asked, it retrieves the most relevant document chunks and provides them as context to the LLM to generate a precise answer.

---

## ✨ Key Features
  * **Chat with Your Data:** Upload one or more PDF documents to create a personalized knowledge base.
  *  **NVIDIA NIM Integration:** Utilizes NVIDIA's high-performance inference endpoints for both state-of-the-art embeddings and the powerful Llama 3.1 70B instruction-tuned model.
  *  **Retrieval-Augmented Generation (RAG):** Ensures answers are grounded in the provided documents, reducing hallucinations and increasing accuracy.
  *  **Dynamic UI:** The user interface, built with Streamlit, is clean, responsive, and updates dynamically as documents are processed.
  *  **Secure API Key Handling:** Uses a `.env` file to manage the NVIDIA API key securely, keeping credentials out of the source code.
  *  **Show Your Work:** Includes an expander to show the exact document chunks that were used as context to generate the answer. 
---
## 🛠️ Requirements

You can install all the necessary Python packages by creating a `requirements.txt` file with the content provided below and running the command:

```bash 
pip install -r requirements.txt
```
---

## 🚀 How to Run

**1. Clone the Repository**
```
git clone https://github.com/syed-masood-pro/NVIDIA_Powered-Q-A.git
cd NVIDIA_Powered-Q-A
```
**2. Set Up Your Environment**

Create a file named `.env` in the root directory of the project. Add your NVIDIA API key to this file:
```
NVIDIA_API_KEY="your_nvidia_api_key_here"
```

You can get a free API key from the [NVIDIA AI Playground](https://build.nvidia.com/meta/llama-3_1-70b-instruct).

**3. Install Dependencies**
```
pip install -r requirements.txt
```

**4. Run the Streamlit App**

From your terminal, run the following command to start the application (assuming your file is named `app.py`):
```bash
streamlit run app.py
```
The app will open in your web browser. Upload your PDFs in the sidebar and start asking questions!

---

## 🖼️ Sample Interface

**Initial View (Ready for Upload)**

(A screenshot of the initial application interface showing the title and the sidebar with the file uploader.)

<img width="959" height="476" alt="proj1" src="https://github.com/user-attachments/assets/83720a24-1f78-43cd-b578-aabdc2b9d69b" />
<img width="959" height="475" alt="proj2" src="https://github.com/user-attachments/assets/e073d830-1ad2-4a04-b26a-e32c32aeb066" />


**Answer with Context**

(A screenshot showing a generated answer with the "Show Document Context" expander open, displaying the source text.)

<img width="959" height="475" alt="pro3" src="https://github.com/user-attachments/assets/78034b7c-d4a5-4d65-a376-5df0b5a764b7" />
<img width="959" height="475" alt="pro4" src="https://github.com/user-attachments/assets/99f31a8d-5c91-47ff-b2c4-9c089ee5f689" />

---

## 📄 License

This project is licensed under the MIT License.

---

## 📧 Contact
**Syed Masood**

✉️ [syedmasood.pro@gmail.com](syedmasood.pro@gmail.com)

🔗 [GitHub Profile](https://github.com/syed-masood-pro/)

💼 [LinkedIn](https://www.linkedin.com/in/syed-masood-pro/)

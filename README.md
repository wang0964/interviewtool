
# InterviewTool – Real-Time Voice Q&A with Local LLM

InterviewTool is a **real-time voice question answering tool** that listens to system audio, converts speech to text, and allows a **local large language model (LLM)** to answer questions instantly.

Press **F12** to send the recognized speech to a local model and receive an AI-generated answer.

---

# Features

- Real-time **system audio listening**
- **Speech-to-text** using Whisper
- **Hotkey trigger (F12)** for instant answering
- Works with **local LLMs via Ollama**
- Fully **offline capable**
- Lightweight and easy to use

---

# How It Works

Workflow:

System Audio → Speech Recognition → Text  
↓  
Press F12  
↓  
Send text to Local LLM (Ollama)  
↓  
Generate Answer  

1. The program captures **system audio**
2. Speech is converted into text using **Whisper**
3. Press **F12** to trigger question answering
4. The recognized text is sent to a **local LLM**
5. The model generates an answer

---

# Local LLM

This project uses **Ollama** to run local language models.

Install Ollama:

https://ollama.com

Recommended models:

- qwen3.5:9b  
- qwen2.5:7b  
- phi3:mini  

Install models:

ollama pull qwen3.5:9b  

---

# Installation

Clone the repository:

git clone https://github.com/wang0964/interviewtool.git  
cd interviewtool  

Install dependencies:

pip install -r requirements.txt  

---

# Run

Start the program:

python audio-new.py  

While the program is running:

Press **F12** → Send the detected speech to the local LLM

Example output:

```
========== [Question] ==========
What is Redis?

========== [Answer] ==========
Redis is an in-memory key-value database commonly used as a cache or message broker.
```

---

# Project Structure
```
interviewtool
│
├── requirements.txt
├── README.md
└── audio-new.py
```
---

# Technologies Used

- Python
- Faster-Whisper
- Ollama
- Local LLM
- WebRTC VAD
- PyTorch
- Rich

---

# Use Cases

- Interview preparation
- Lecture listening
- Meeting assistance
- Real-time AI assistant

---

# Requirements

- Python 3.10+
- GPU recommended (RTX series, VRAM >= 8Gb)
- Ollama installed
- RAM >= 16Gb

---

# License

MIT License

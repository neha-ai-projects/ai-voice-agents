# Customer Support Voice Agent 🎙️

A **Streamlit-based application** that provides **voice-powered answers to documentation questions**. This agent crawls documentation websites, stores the information in a vector database, and answers user queries using AI models with speech synthesis capabilities.

---

## 🚀 Features

- 🌐 **Documentation web crawling** using [Firecrawl](https://firecrawl.dev)
- 🧠 **Vector search** powered by [Qdrant](https://qdrant.tech)
- 🤖 **AI response generation** (OpenAI GPT-4o or Replicate LLaMA-3)
- 🎤 **Voice input** capability
- 🔊 **Text-to-speech output** with Google TTS
- 🔍 **Source citation** from documentation

---

## 📦 Requirements

```
streamlit
requests
uuid
speech_recognition
replicate
gtts
qdrant_client
fastembed
firecrawl
```

---

## 🔐 API Keys Required

- **Qdrant**: For vector database (URL and API key)
- **Firecrawl**: For web crawling documentation sites
- **OpenAI** or **Replicate**: For AI text generation

---

## ⚙️ Setup & Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 🎯 Usage

1. Configure your API keys in the **sidebar**
2. Enter the documentation URL you want to crawl and learn from
3. Click **Initialize System** (to crawl and index docs)
4. Ask questions via **text input** or **voice recording**
5. Get **AI-generated answers** with audio playback

---

## 🛠️ Configuration Options

### 🧠 Text Model Selection
Choose between:
- OpenAI's **GPT-4o**
- Replicate's **LLaMA-3 (70B Instruct)**

### 🔊 Voice Settings
- Multiple **language options** for TTS
- Adjustable **speech rate**
- **Microphone selection** for input

---

## 🔄 How It Works

1. **Crawling**: Uses **Firecrawl** to gather documentation content from a URL
2. **Embedding**: Content is converted into vector embeddings using **FastEmbed**
3. **Storage**: Embeddings are stored in the **Qdrant** vector database
4. **Query Processing**: 
   - User query is embedded
   - Relevant docs are retrieved from Qdrant
   - AI generates response using selected model
5. **Voice Output**: Response is converted to speech via **Google TTS**

---

## 🗂️ Project Structure

- `qdrant_integration.py`: Vector database setup and querying
- `firecrawl_integration.py`: Documentation crawling logic
- `ai_model.py`: OpenAI and Replicate model integration
- `voice_processing.py`: Speech recognition and TTS logic
- `app.py`: Streamlit UI and main app logic

---

## 📄 License

[MIT License](LICENSE)

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io) – for the interactive UI
- [Qdrant](https://qdrant.tech) – for vector search
- [Firecrawl](https://firecrawl.dev) – for documentation crawling
- [OpenAI](https://openai.com) and [Replicate](https://replicate.com) – for AI models
- [Google TTS](https://cloud.google.com/text-to-speech) – for speech synthesis

from typing import List, Dict, Optional
import requests
import uuid
import os
import tempfile
import time
from datetime import datetime
import asyncio
import streamlit as st
import speech_recognition as sr
import replicate
from gtts import gTTS
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from fastembed import TextEmbedding
from firecrawl import FirecrawlApp
from agents import Agent, Runner

def setup_qdrant_collection(qdrant_url: str, qdrant_api_key: str, collection_name: str = "docs_embeddings"):
    """Setup Qdrant vector database connection and collection"""
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    embedding_model = TextEmbedding()
    test_embedding = list(embedding_model.embed(["test"]))[0]
    embedding_dim = len(test_embedding)
    
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )
    except Exception as e:
        if "already exists" not in str(e):
            raise e
    
    return client, embedding_model

def crawl_documentation(firecrawl_api_key: str, url: str, output_dir: Optional[str] = None):
    """Crawl documentation from provided URL using Firecrawl"""
    firecrawl = FirecrawlApp(api_key=firecrawl_api_key)
    pages = []
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    response = firecrawl.crawl_url(
        url,
        params={
            'limit': 5,
            'scrapeOptions': {
                'formats': ['markdown', 'html']
            }
        }
    )
    
    while True:
        for page in response.get('data', []):
            content = page.get('markdown') or page.get('html', '')
            metadata = page.get('metadata', {})
            source_url = metadata.get('sourceURL', '')
            
            if output_dir and content:
                filename = f"{uuid.uuid4()}.md"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            pages.append({
                "content": content,
                "url": source_url,
                "metadata": {
                    "title": metadata.get('title', ''),
                    "description": metadata.get('description', ''),
                    "language": metadata.get('language', 'en'),
                    "crawl_date": datetime.now().isoformat()
                }
            })
        
        next_url = response.get('next')
        if not next_url:
            break
            
        response = firecrawl.get(next_url)
        time.sleep(1)
    
    return pages

def store_embeddings(client: QdrantClient, embedding_model: TextEmbedding, pages: List[Dict], collection_name: str):
    """Store document embeddings in Qdrant collection"""
    for page in pages:
        embedding = list(embedding_model.embed([page["content"]]))[0]
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "content": page["content"],
                        "url": page["url"],
                        **page["metadata"]
                    }
                )
            ]
        )

def init_session_state():
    """Initialize session state with default values"""
    defaults = {
        "initialized": False,
        "qdrant_url": "",
        "qdrant_api_key": "",
        "firecrawl_api_key": "",
        "openai_api_key": "",
        "replicate_api_key": "",
        "doc_url": "",
        "setup_complete": False,
        "client": None,
        "embedding_model": None,
        "processor_agent": None,
        "selected_voice": "en",  # default voice for gTTS
        "text_model_choice": "OpenAI",
        "enable_voice_input": False,
        "tts_voice_speed": 1.0  # default speed for TTS
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def generate_speech_with_gtts(text, lang="en", slow=False):
    """Generate speech using Google Text-to-Speech (gTTS)"""
    try:
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"response_{uuid.uuid4()}.mp3")
        
        # Generate TTS using gTTS
        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save(audio_path)
        
        return audio_path
    except Exception as e:
        raise Exception(f"gTTS Error: {str(e)}")

def setup_voice_input():
    """Set up voice input configuration in the sidebar"""
    st.sidebar.subheader("Voice Input Configuration")

    # Get a list of available microphones
    mic_list = sr.Microphone.list_microphone_names()
    
    if mic_list:
        mic_index = st.sidebar.selectbox(
            "Select Microphone", 
            range(len(mic_list)), 
            format_func=lambda x: mic_list[x]
        )
        st.session_state.selected_mic_index = mic_index
    else:
        st.sidebar.warning("No microphones found. Please check your device.")

    # Option to enable or disable voice input
    st.session_state.enable_voice_input = st.sidebar.checkbox(
        "Enable Voice Input", 
        value=st.session_state.get("enable_voice_input", False)
    )

    # Check and display microphone selection status
    if 'selected_mic_index' in st.session_state:
        st.sidebar.success(f"Microphone '{mic_list[st.session_state.selected_mic_index]}' selected.")
    else:
        st.sidebar.warning("Please select a microphone for voice input.")

def capture_voice_input():
    """Capture voice input and convert it into text."""
    if st.session_state.get("enable_voice_input", False) and 'selected_mic_index' in st.session_state:
        recognizer = sr.Recognizer()
        try:
            with st.spinner("🎤 Listening..."):
                with sr.Microphone(device_index=st.session_state.selected_mic_index) as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source, timeout=5)
                try:
                    query = recognizer.recognize_google(audio)
                    st.success(f"You said: {query}")
                    return query
                except sr.UnknownValueError:
                    st.error("Sorry, I could not understand the audio.")
                except sr.RequestError:
                    st.error("Could not request results from speech recognition service.")
        except Exception as e:
            st.error(f"Error with microphone: {str(e)}")
    return None

def sidebar_config():
    """Configure sidebar with API keys and settings"""
    with st.sidebar:
        st.title("🔑 Configuration")
        st.markdown("---")

        st.session_state.qdrant_url = st.text_input("Qdrant URL", value=st.session_state.qdrant_url, type="password")
        st.session_state.qdrant_api_key = st.text_input("Qdrant API Key", value=st.session_state.qdrant_api_key, type="password")
        st.session_state.firecrawl_api_key = st.text_input("Firecrawl API Key", value=st.session_state.firecrawl_api_key, type="password")

        st.markdown("### 🤖 Text Model")
        st.session_state.text_model_choice = st.radio(
            "Choose text model",
            options=["OpenAI", "Replicate"],
            index=["OpenAI", "Replicate"].index(st.session_state.text_model_choice)
        )

        if st.session_state.text_model_choice == "OpenAI":
            st.session_state.openai_api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
        else:
            st.session_state.replicate_api_key = st.text_input("Replicate API Key", value=st.session_state.replicate_api_key, type="password")

        st.markdown("---")
        st.session_state.doc_url = st.text_input("Documentation URL", value=st.session_state.doc_url, placeholder="https://docs.example.com")

        st.markdown("---")
        st.markdown("### 🔊 Google TTS Settings")
        
        # Voice language selection for gTTS
        voice_options = {
            "en": "English",
            "fr": "French",
            "es": "Spanish",
            "de": "German",
            "it": "Italian",
            "ja": "Japanese",
            "ko": "Korean",
            "pt": "Portuguese",
            "ru": "Russian",
            "zh-CN": "Chinese (Simplified)"
        }
        
        st.session_state.selected_voice = st.selectbox(
            "Select TTS Language", 
            options=list(voice_options.keys()),
            format_func=lambda x: voice_options[x],
            index=list(voice_options.keys()).index(st.session_state.selected_voice)
        )
        
        # Speech speed option
        st.session_state.tts_voice_slow = st.checkbox(
            "Slower speech rate", 
            value=False,
            help="Enable slower speech rate for better clarity"
        )
        
        # Add voice input configuration
        setup_voice_input()

        if st.button("Initialize System", type="primary"):
            required_fields_filled = all([
                st.session_state.qdrant_url,
                st.session_state.qdrant_api_key,
                st.session_state.firecrawl_api_key,
                st.session_state.doc_url,
                (st.session_state.openai_api_key if st.session_state.text_model_choice == "OpenAI" 
                 else st.session_state.replicate_api_key)
            ])

            if required_fields_filled:
                progress_placeholder = st.empty()
                with progress_placeholder.container():
                    try:
                        st.markdown("🔄 Setting up Qdrant connection...")
                        client, embedding_model = setup_qdrant_collection(
                            st.session_state.qdrant_url,
                            st.session_state.qdrant_api_key
                        )
                        st.session_state.client = client
                        st.session_state.embedding_model = embedding_model
                        st.markdown("✅ Qdrant setup complete!")

                        st.markdown("🔄 Crawling documentation pages...")
                        pages = crawl_documentation(
                            st.session_state.firecrawl_api_key,
                            st.session_state.doc_url
                        )
                        st.markdown(f"✅ Crawled {len(pages)} documentation pages!")

                        st.markdown("🔄 Storing embeddings...")
                        store_embeddings(client, embedding_model, pages, "docs_embeddings")
                        st.markdown("✅ Embeddings stored successfully!")

                        # Use selected model for processor_agent
                        if st.session_state.text_model_choice == "OpenAI":
                            st.session_state.processor_agent = setup_openai_agent(st.session_state.openai_api_key)
                        else:
                            st.session_state.processor_agent = setup_llama_agent(st.session_state.replicate_api_key)

                        st.session_state.setup_complete = True
                        st.success("✅ System initialized successfully!")

                    except Exception as e:
                        st.error(f"Error during setup: {str(e)}")
            else:
                st.error("Please fill in all the required fields!")

def setup_openai_agent(openai_api_key: str):
    """Set up OpenAI agent"""
    os.environ["OPENAI_API_KEY"] = openai_api_key

    processor_agent = Agent(
        name="Documentation Processor",
        instructions="""You are a helpful documentation assistant. Your task is to:
        1. Analyze the provided documentation content
        2. Answer the user's question clearly and concisely
        3. Include relevant examples when available
        4. Cite the source URLs when referencing specific content
        5. Keep responses natural and conversational
        6. Format your response in a way that's easy to speak out loud""",
        model="gpt-4o"
    )

    return processor_agent

def setup_llama_agent(replicate_api_key: str):
    """Set up the LLaMA agent using Replicate"""
    os.environ["REPLICATE_API_KEY"] = replicate_api_key

    # Initialize the Replicate client with the provided API key
    replicate.Client(api_token=replicate_api_key)

    # Return LLaMA agent
    llama_agent = Agent(
        name="LLaMA Documentation Processor",
        instructions="""You are a LLaMA-based documentation assistant. Your task is to:
        1. Analyze the provided documentation content
        2. Answer the user's question clearly and concisely
        3. Include relevant examples when available
        4. Cite the source URLs when referencing specific content
        5. Keep responses natural and conversational
        6. Format your response in a way that's easy to speak out loud""",
        model="replicate/llama-70b-instruct"
    )
    
    return llama_agent

async def process_query(
    query: str,
    client: QdrantClient,
    embedding_model: TextEmbedding,
    processor_agent,
    collection_name: str = "docs_embeddings",
    text_model_choice: str = "OpenAI",
    openai_api_key: str = "",
    replicate_api_key: str = "",
    tts_voice: str = "en",
    tts_slow: bool = False
):
    """Process user query, search documentation, generate response and TTS"""
    try:
        # Generate embedding for the query
        query_embedding = list(embedding_model.embed([query]))[0]
        
        # Search for relevant documents
        search_response = client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            limit=3,
            with_payload=True
        )

        search_results = search_response.points if hasattr(search_response, 'points') else []

        if not search_results:
            raise Exception("No relevant documents found in the vector database")

        # Prepare context for the LLM
        context = "Based on the following documentation:\n\n"
        for result in search_results:
            payload = result.payload
            if not payload:
                continue
            url = payload.get('url', 'Unknown URL')
            content = payload.get('content', '')
            context += f"From {url}:\n{content}\n\n"

        context += f"\nUser Question: {query}\n\n"
        context += "Please provide a clear, concise answer that can be easily spoken out loud."

        # Generate text response based on selected model
        processor_response = ""
        
        if text_model_choice == "OpenAI":
            # Use OpenAI model via Agent/Runner
            processor_result = await Runner.run(processor_agent, context)
            processor_response = processor_result.final_output
        else:
            try:
        # Initialize Replicate client
                os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
                client = replicate.Client(api_token=replicate_api_key)
        
        # Run the model
                print("🔍 LLM CONTEXT:\n", context)
                output = replicate.run("meta/meta-llama-3-70b-instruct",
                input={
                "top_p": 0.9,
                "prompt": context,
                "max_length": 1024,
                "temperature": 0.6,
                "presence_penalty": 1.15
            }
        )
                 # For debugging
                print(f"Replicate API response type: {type(output)}")
                print(f"Output", output)
        
        # Handle different response formats
                if isinstance(output, str):
                    processor_response = output
                elif hasattr(output, '__iter__'):
                    processor_response = "".join([str(chunk) for chunk in output])
                else:
                    processor_response = str(output)
                print(f"Response:", processor_response)
        # # Concatenate output chunks
        #         processor_response = "".join([chunk for chunk in output])
            except Exception as e:
                print(f"Detailed error with Replicate: {str(e)}")
                if hasattr(e, '__dict__'):
                    print(f"Error attributes: {e.__dict__}")
                raise Exception(f"Error generating LLaMA response: {str(e)}")

        # Generate audio from text using Google TTS
        audio_path = generate_speech_with_gtts(
            text=processor_response,
            lang=tts_voice,
            slow=tts_slow
        )

        return {
            "status": "success",
            "text_response": processor_response,
            "audio_path": audio_path,
            "sources": [r.payload.get("url", "Unknown URL") for r in search_results if r.payload],
            "query_details": {
                "vector_size": len(query_embedding),
                "results_found": len(search_results),
                "collection_name": collection_name
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "query": query
        }
    
def run_streamlit():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Customer Support Voice Agent",
        page_icon="🎙️",
        layout="wide"
    )
    
    init_session_state()
    sidebar_config()
    
    st.title("🎙️ Customer Support Voice Agent")
    st.markdown("""  
    Get voice-powered answers to your documentation questions! Simply:
    1. Configure your API keys in the sidebar
    2. Enter the documentation URL you want to learn about
    3. Ask your question via text or voice input
    """)
    
    # Create columns for text input and voice input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "What would you like to know about the documentation?",
            placeholder="e.g., How do I authenticate API requests?",
            disabled=not st.session_state.setup_complete
        )
    
    with col2:
        if st.session_state.setup_complete and st.session_state.get("enable_voice_input", False):
            if st.button("🎤 Record Question", disabled=not st.session_state.setup_complete):
                voice_query = capture_voice_input()
                if voice_query:
                    query = voice_query
    
    # Process query if available
    if query and st.session_state.setup_complete:
        with st.status("Processing your query...", expanded=True) as status:
            try:
                st.markdown("🔄 Searching documentation and generating response...")
                
                # Process query with appropriate API keys
                result = asyncio.run(process_query(
                    query=query,
                    client=st.session_state.client,
                    embedding_model=st.session_state.embedding_model,
                    processor_agent=st.session_state.processor_agent,
                    collection_name="docs_embeddings",
                    text_model_choice=st.session_state.text_model_choice,
                    openai_api_key=st.session_state.openai_api_key,
                    replicate_api_key=st.session_state.replicate_api_key,
                    tts_voice=st.session_state.selected_voice,
                    tts_slow=st.session_state.tts_voice_slow
                ))
                
                if result["status"] == "success":
                    status.update(label="✅ Query processed!", state="complete")
                    
                    st.markdown("### Response:")
                    st.write(result["text_response"])
                    
                    if "audio_path" in result and result["audio_path"]:
                        voice_options = {
                            "en": "English",
                            "fr": "French",
                            "es": "Spanish",
                            "de": "German",
                            "it": "Italian",
                            "ja": "Japanese",
                            "ko": "Korean",
                            "pt": "Portuguese",
                            "ru": "Russian",
                            "zh-CN": "Chinese (Simplified)"
                        }
                        language_name = voice_options.get(st.session_state.selected_voice, "Unknown")
                        
                        st.markdown(f"### 🔊 Audio Response (Language: {language_name})")
                        st.audio(result["audio_path"], format="audio/mp3", start_time=0)
                        
                        with open(result["audio_path"], "rb") as audio_file:
                            audio_bytes = audio_file.read()
                            st.download_button(
                                label="📥 Download Audio Response",
                                data=audio_bytes,
                                file_name=f"voice_response_{st.session_state.selected_voice}.mp3",
                                mime="audio/mp3"
                            )
                    
                    st.markdown("### Sources:")
                    for source in result["sources"]:
                        st.markdown(f"- {source}")
                else:
                    status.update(label="❌ Error processing query", state="error")
                    st.error(f"Error: {result.get('error', 'Unknown error occurred')}")
                    
            except Exception as e:
                status.update(label="❌ Error processing query", state="error")
                st.error(f"Error processing query: {str(e)}")
    
    elif not st.session_state.setup_complete:
        st.info("👈 Please configure the system using the sidebar first!")

if __name__ == "__main__":
    run_streamlit()
import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
import time
from audio_recorder_streamlit import audio_recorder
from RAG_functions import generate_answer, assistant

# Load environment variables from .env file
load_dotenv()
genai.configure(api_key=os.environ["API_KEY"])

st.set_page_config(
    page_title="Recipe Guide Chatbot",
    page_icon="🍕",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.title("🍕 Recipe Guide Chatbot 🥙")


# Reset the conversation with the chatbot
def reset_conv():
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        st.session_state.pop("messages", None)


# Function to handle audio input
def handle_audio():
    audio_prompt = None
    if "prev_speech_hash" not in st.session_state:
        st.session_state.prev_speech_hash = None

    if speech_input and st.session_state.prev_speech_hash != hash(speech_input):
        st.session_state.prev_speech_hash = hash(speech_input)
        audio_file_path = "audio_input.mp3"
        with open(audio_file_path, "wb") as f:
            f.write(speech_input)
        myfile = genai.upload_file(path=audio_file_path)

        prompt = "Convert speech to text"

        # Pass the prompt and the uploaded file to Gemini for transcription
        response = assistant.generate_content([prompt, myfile])

        # Get the transcription result from the response
        audio_prompt = response.text
        return audio_prompt
    return None


# Function to handle the bot's response
def handle_response(prompt):
    stream = generate_answer(prompt)

    # show the bot's response in chunks (streaming the answer)
    chat_message = ""
    stream.resolve()
    placeholder = st.empty()
    chunk_size = 10  # size of each chunk of characters to be displayed at a time
    buffer = ""
    for chunk in stream:
        buffer += chunk.text  # append chunk text to the buffer
        # split the buffer into chunks
        for i in range(0, len(buffer), chunk_size):
            # append the next 10 characters to the chat_message
            chat_message += buffer[i : i + chunk_size]
            placeholder.markdown(
                f"<div class='bot-message'>{chat_message}</div>",
                unsafe_allow_html=True,
            )
            time.sleep(0.01)  # delay between chunks
        buffer = ""
    return chat_message


with st.sidebar:
    st.markdown(
        """
        <h1 style='text-align: center;'>🍕 Recipe Guide Chatbot 🥙</h1>
        <p style='text-align: center;'>Ask me anything about recipes!</p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    speech_input = audio_recorder(
        "Talk to the chatbot:",
        icon_size="2x",
        neutral_color="#2C6FC3",
    )
    audio_prompt = handle_audio()

    st.button("Reset Conversation", on_click=reset_conv)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input and generate answer
if prompt := st.chat_input("What recipe are you looking for today?") or audio_prompt:
    # Add user message to chat history
    st.session_state.messages.append(
        {"role": "user", "content": prompt or audio_prompt}
    )
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(
            f"<div class='user-message'>{prompt}</div>",
            unsafe_allow_html=True,
        )

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        assistant_message = handle_response(prompt)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_message}
    )

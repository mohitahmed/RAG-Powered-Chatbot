# recipe-guide-LLM
##About
A retrieval augmented generation powered recipe chatbot using Google Gemini and streamlit for user interface.

## Prerequisites

1. Create a virtual environment:

`python -m venv venv`

2. Activate the virtual environment:

`source venv/bin/activate`

3. Install required packages

`pip install -r requirements.txt`

## Setting Up Environment Variables

1. Create a .env file in your project root directory:
   `touch .env`

2. Add your API keys to the .env file. Open the file in a text editor and add the following lines:
   `API_KEY=your_gemini_api_key_here`

## Running the Chatbot

To run the Streamlit chatbot application, use the following command:
`streamlit run chatbot.py`

## Final Chatbot

The final codes for the chatbot are available in 'RAG_functions.py' and 'chatbot.py' files.

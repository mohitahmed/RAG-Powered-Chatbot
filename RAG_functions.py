import numpy as np
import pandas as pd
from torch import torch
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai
import re

# Initialize the GenerativeAI client
assistant = genai.GenerativeModel("gemini-1.5-flash")
chat = assistant.start_chat()


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


# Function to clean up the string representation of embeddings
def clean_embedding_string(embedding_str):
    # Remove any unwanted characters like newlines or extra spaces
    cleaned_str = embedding_str.replace("\n", " ").strip()

    # Add commas between numbers if they are missing
    cleaned_str = re.sub(r"\s+", ", ", cleaned_str)

    # Remove outer brackets
    cleaned_str = cleaned_str.replace("[", "").replace("]", "")

    # Split the cleaned string by commas to get the values
    try:
        values = [float(x) for x in cleaned_str.split(",") if x.strip()]
        array_2d = np.array(values).reshape(1, -1)
    except ValueError as e:
        print(f"Error parsing the string: {embedding_str}")
        array_2d = np.array([])  # Return an empty array in case of failure

    return array_2d


# Load the CSV file and apply the cleaning function
embeddings_df = pd.read_csv("embeddings_df.csv")

# Apply the cleaning function to the 'embeddings' column
embeddings_df["embeddings"] = embeddings_df["embeddings"].apply(clean_embedding_string)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Function to get embeddings from the model for a given text
def get_embeddings(text):
    # Tokenize the input text and move it to the GPU
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
        device
    )
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the embeddings (usually the last hidden state)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings


# Function to retrieve the top N most relevant documents based on cosine similarity between the user query and document embeddings
def get_relevant_docs(user_query, embeddings_df, top_n=3):
    query_embeddings = np.array(get_embeddings(user_query))

    def cosine_similarity(embedding):
        return float(
            np.dot(query_embeddings, embedding)
            / (np.linalg.norm(query_embeddings) * np.linalg.norm(embedding))
        )

    embeddings_df["similarity"] = embeddings_df["embeddings"].apply(
        lambda x: cosine_similarity(np.array(x)[0])
    )

    relevant_docs = embeddings_df.nlargest(top_n, "similarity")["input"].tolist()

    return relevant_docs


def make_rag_prompt(query, relevant_passage):
    # Ensure all elements in relevant_passage are strings before joining
    relevant_passage = " ".join([str(passage) for passage in relevant_passage])

    prompt = (
        f"You are a helpful and friendly recipe chatbot. Answer questions using the text from the reference passage below.\n\n"
        f"Focus only on topics related to recipes, ingredients, directions, or cooking methods.\n"
        f"If the question asks for a recipe that is not in the passage, acknowledge that you do not know.\n"
        f"Provide extra details to make your response more engaging.\n"
        f"Ignore the question if the passage is irrelevant.\n"
        f"Format the answer in markdown. If listing ingredients, use an unordered list with a title.\n\n"
        f"Do not mention the passage in the answer.\n\n"
        f"QUESTION: '{query}'\n"
        f"PASSAGE: '{relevant_passage}'\n\n"
        f"ANSWER:"
    )
    return prompt


# Function to generate a response and return the bot's answer
def generate_response(chat, user_prompt):
    answer = chat.send_message(
        user_prompt,
        stream=True,
    )
    return answer


def generate_answer(query):
    relevant_text = get_relevant_docs(query, embeddings_df, 3)
    prompt = make_rag_prompt(query, relevant_passage=relevant_text)
    answer = generate_response(chat, prompt)
    return answer

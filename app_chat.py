import streamlit as st
import pandas as pd
from groq import Groq
import pinecone
from openai.embeddings_utils import get_embedding

# Initialize Pinecone
pinecone.init(api_key="ed2e35ad-250c-4831-a198-229d14c0901d", environment="us-west1-gcp")

# Create or connect to Pinecone index
index_name = "fashion-assistant-conversations"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)  # Using OpenAI text-embedding-ada-002

index = pinecone.Index(index_name)

# Initialize the Groq client with your API key
client = Groq(api_key="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")

# Define the system message for the model
system_message = {
    "role": "system",
    "content": "You are an experienced Fashion designer who starts conversation with proper greet, giving valuable and catchy fashion advices and suggestions, stays to the point, and asks questions only if the user has any concerns. You take inputs like name, age, gender, location, ethnicity, height, weight, skin tone."
}

# Function to reset the chat
def reset_chat(user_id):
    st.session_state.messages = []
    st.session_state.chat_title = "New Chat"
    # Optionally, clear the user's conversation history from Pinecone
    index.delete(ids=[f"{user_id}-*"])  # Deletes all conversations associated with the user

# Function to store messages in Pinecone
def store_message_in_pinecone(message, user_id, role):
    # Create an embedding for the message
    embedding = get_embedding(message, engine="text-embedding-ada-002")
    
    # Store the message and its embedding in Pinecone
    index.upsert(vectors=[(f"{user_id}-{len(st.session_state.messages)}", embedding, {"role": role, "content": message})])

# Function to retrieve conversation history from Pinecone
def retrieve_conversation_from_pinecone(user_id):
    query_response = index.query(f"{user_id}*", top_k=100, include_metadata=True)
    conversation = [{"role": item['metadata']['role'], "content": item['metadata']['content']} for item in query_response['matches']]
    return conversation

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_title = "Fashion Assistant"

user_id = st.text_input("Enter your user ID:", value="user_1")  # Unique user identifier

# Check if there's existing conversation in Pinecone and retrieve it
if len(st.session_state.messages) == 0:
    previous_conversation = retrieve_conversation_from_pinecone(user_id)
    if previous_conversation:
        st.session_state.messages = previous_conversation

# Sidebar for user inputs
with st.sidebar:
    st.header("User Inputs")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    location = st.text_input("Location")
    gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
    ethnicity = st.selectbox("Ethnicity", options=["Asian", "Black", "Hispanic", "White", "Other"])
    height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
    skin_tone = st.text_input("Skin Tone (Hex Code)", value="#ffffff")

    if st.button("Reset Chat"):
        reset_chat(user_id)

# Chat UI
st.title(st.session_state.chat_title)

# Display all previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for the chat
user_input = st.chat_input("Ask anything about fashion...")
if user_input:
    # Store user message in the chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    store_message_in_pinecone(user_input, user_id, "user")

    # Prepare messages for the API call, including the previous conversation
    messages = [system_message] + st.session_state.messages

    try:
        # Generate a response from the Groq API
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,  # Send the entire conversation
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
        )

        # Ensure response is valid
        if completion.choices and len(completion.choices) > 0:
            response_content = completion.choices[0].message.content
        else:
            response_content = "Sorry, I couldn't generate a response."

    except Exception as e:
        response_content = f"Error: {str(e)}"

    # Store assistant response in the chat history
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    store_message_in_pinecone(response_content, user_id, "assistant")

    # Display assistant response
    with st.chat_message(name="assistant"):
        st.markdown(response_content)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

# Load Hugging Face API token
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if not api_token:
    raise ValueError("❌ Missing HUGGINGFACEHUB_API_TOKEN in .env file!")

# Choose a working open model (no gating required)
repo_id = "HuggingFaceH4/zephyr-7b-beta"   # You can also try "mistralai/Mistral-7B-Instruct-v0.2"

# Create LLM endpoint
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token=api_token,
    temperature=0.7,
    max_new_tokens=256
)

# Wrap in Chat interface
model = ChatHuggingFace(llm=llm)

# System instruction (keeps answers short + focused)
system_message = SystemMessage(
    content="You are a helpful AI assistant. Always respond to the user's latest input only. "
            "Do not roleplay as multiple characters. Keep responses clear and concise."
)

# Conversation state
messages = [system_message]
chat_history = []

# Maximum messages to send back each turn (avoid long transcripts)
MAX_CONTEXT = 6

print("🤖 Chatbot ready! Type 'exit' or 'q' to quit.\n")

# Chat loop
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "q"]:
        print("Exiting chatbot... 👋")
        break

    # Add latest user message
    messages.append(HumanMessage(content=user_input))

    # Keep context trimmed
    context = [system_message] + messages[-MAX_CONTEXT:]

    # Generate response
    result = model.invoke(context)

    # Save and show reply
    messages.append(AIMessage(content=result.content))
    chat_history.append({"User": user_input, "AI": result.content})

    print("AI:", result.content)

# Optional: save conversation to file
# import json
# with open("chat_history.json", "w", encoding="utf-8") as f:
#     json.dump(chat_history, f, indent=2, ensure_ascii=False)

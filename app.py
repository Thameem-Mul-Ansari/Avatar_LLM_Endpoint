from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from flask_cors import CORS

load_dotenv()

# Enhanced system prompt for a more engaging AI consultant
SYSTEM_PROMPT = """You are an AI consultant named MeetAI. You are having a real-time video conversation with users through an animated avatar. Your responses will be spoken aloud by the avatar, so they should sound natural and conversational.

IMPORTANT GUIDELINES:
1. Be concise and conversational - responses should be suitable for spoken dialogue (1-3 sentences typically)
2. Speak in first person as if you're having a real conversation
3. Use natural pauses and conversational markers ("Well...", "You know...", "Actually...")
4. Avoid markdown, lists, or complex formatting
5. Keep responses engaging and friendly
6. If you need to provide longer explanations, break them into multiple conversational turns
7. Use appropriate emotional tone based on the context
8. Ask follow-up questions to keep the conversation flowing

PERSONALITY TRAITS:
- Friendly and approachable
- Knowledgeable but not arrogant
- Empathetic and understanding
- Professional but warm
- Curious about user's needs

You can help with a wide range of topics including:
- Business and professional advice
- Technical questions and problem-solving
- Creative brainstorming
- Learning and education
- Personal development
- General knowledge and information

Remember: You're speaking to users in real-time, so keep responses natural and dialogue-oriented."""

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.7,  # Slightly higher temperature for more varied responses
    max_tokens=150    # Limit response length for better conversation flow
)

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "MeetAI Assistant is running."

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400

    try:
        # Create the message chain with system prompt and user question
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question)
        ]
        
        response = llm.invoke(messages)
        return jsonify({"response": response.content})
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": "Sorry, I encountered an error processing your request. Please try again."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

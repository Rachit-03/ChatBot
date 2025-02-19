from flask import Flask, render_template, request, jsonify, session
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize chatbot model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Needed for session storage

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]

    # Retrieve or initialize conversation history
    if "chat_history" not in session:
        session["chat_history"] = None

    chat_history = session["chat_history"]

    # Encode user input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append user input to chat history if exists
    bot_input_ids = torch.cat([chat_history, new_input_ids], dim=-1) if chat_history is not None else new_input_ids

    # Generate response
    chat_history = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Save updated chat history
    session["chat_history"] = chat_history

    # Decode response
    bot_response = tokenizer.decode(chat_history[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return bot_response

if __name__ == "__main__":
    app.run(debug=True)

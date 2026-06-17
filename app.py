"""
app.py — Flask web application for the domain-specific GPT chat interface.

Usage:
    python app.py
    python app.py --model output/model --port 5000

Endpoints:
    GET  /        — Serves the chat UI (templates/index.html)
    POST /chat    — Accepts JSON { "question": "..." }, returns { "response": "..." }
    GET  /health  — Health check, returns model status
"""

import os
import argparse
from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Global model state
model = None
tokenizer = None
MODEL_PATH = os.environ.get("MODEL_PATH", "output/model")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "200"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))


def load_model(model_path: str):
    global model, tokenizer
    print(f"Loading model from: {model_path}")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load fine-tuned model ({e}). Falling back to base gpt2.")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()


def generate(question: str, max_length: int = None, temperature: float = None) -> str:
    if model is None or tokenizer is None:
        return "Model is not loaded. Please train the model first."

    max_length = max_length or MAX_LENGTH
    temperature = temperature or TEMPERATURE

    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in decoded:
        answer = decoded.split("Answer:", 1)[1].strip()
        if "Question:" in answer:
            answer = answer.split("Question:")[0].strip()
        return answer

    return decoded.strip()


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Field 'question' is required and must not be empty."}), 400
    if len(question) > 500:
        return jsonify({"error": "Question must be 500 characters or fewer."}), 400

    max_length = data.get("max_length", MAX_LENGTH)
    temperature = data.get("temperature", TEMPERATURE)

    try:
        response = generate(question, max_length=max_length, temperature=temperature)
        return jsonify({"question": question, "response": response})
    except Exception as e:
        app.logger.error(f"Generation error: {e}")
        return jsonify({"error": "Failed to generate response. See server logs."}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    })


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run the GPT chat Flask web app.")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to fine-tuned model directory.")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    args = parser.parse_args()

    global MODEL_PATH
    MODEL_PATH = args.model

    load_model(MODEL_PATH)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

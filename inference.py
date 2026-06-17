"""
inference.py — Load a fine-tuned GPT-2 model and generate domain-specific responses.

Usage:
    python inference.py
    python inference.py --model output/model --max_length 200
    python inference.py --question "What is machine learning?"
"""

import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def load_model(model_path: str):
    print(f"Loading model from: {model_path}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    question: str,
    max_length: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    num_return_sequences: int = 1,
) -> str:
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the answer portion
    if "Answer:" in decoded:
        answer = decoded.split("Answer:", 1)[1].strip()
        # Stop at the next "Question:" if present
        if "Question:" in answer:
            answer = answer.split("Question:")[0].strip()
        return answer
    return decoded.strip()


def interactive_mode(model, tokenizer, max_length: int):
    print("\nDomain-Specific GPT — Interactive Mode")
    print("Type your question and press Enter. Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue
        response = generate_response(model, tokenizer, question, max_length=max_length)
        print(f"GPT: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate responses using a fine-tuned GPT-2 model.")
    parser.add_argument("--model", type=str, default="output/model", help="Path to fine-tuned model directory.")
    parser.add_argument("--question", type=str, default=None, help="Single question to answer (skips interactive mode).")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum token length of generated response.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0.0–1.0).")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling probability.")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.question:
        print(f"\nQuestion: {args.question}")
        response = generate_response(
            model, tokenizer, args.question,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"Answer: {response}")
    else:
        interactive_mode(model, tokenizer, max_length=args.max_length)


if __name__ == "__main__":
    main()

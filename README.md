# Customized GPT — Domain-Specific Fine-Tuned Language Model

A fine-tuned GPT-2 model trained on domain-specific data to generate intelligent, contextually relevant responses. Includes a training pipeline, a console inference tool, and a Flask web chat interface.

---

## Project Description

Generic language models often give vague or inaccurate answers for specialized domains. This project fine-tunes GPT-2 on a curated dataset of domain-specific question-answer pairs (machine learning concepts by default), producing a model that gives focused, accurate responses within that domain.

**What's included:**
- `train.py` — fine-tune GPT-2 on your own text data
- `inference.py` — interactive console Q&A with the fine-tuned model
- `app.py` — Flask web app with a clean chat UI
- `data/train.txt` — 20 curated ML domain Q&A pairs (replace with your own)

---

## Project Structure

```
customized-gpt-domain-specific/
├── train.py              # Fine-tuning script
├── inference.py          # Console inference / interactive mode
├── app.py                # Flask web application
├── templates/
│   └── index.html        # Chat UI
├── data/
│   └── train.txt         # Domain-specific training data (Q&A pairs)
├── output/
│   └── model/            # Fine-tuned model saved here (after training)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Althiushasree/customized-gpt-domain-specific.git
cd customized-gpt-domain-specific
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3. Install PyTorch

Visit [pytorch.org](https://pytorch.org/get-started/locally/) and pick the right command for your OS and CUDA version. Example for CPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install other requirements

```bash
pip install -r requirements.txt
```

---

## Training the Model

### Prepare your data

Edit `data/train.txt` with your own domain Q&A pairs in the format:

```
Question: What is X?
Answer: X is ...

Question: How does Y work?
Answer: Y works by ...
```

### Run training

```bash
python train.py
```

With custom arguments:

```bash
python train.py \
  --data data/train.txt \
  --output output/model \
  --model gpt2 \
  --epochs 3 \
  --batch_size 2 \
  --lr 5e-5
```

| Argument | Default | Description |
|---|---|---|
| `--data` | `data/train.txt` | Path to training text file |
| `--output` | `output/model` | Directory to save the fine-tuned model |
| `--model` | `gpt2` | Base model (`gpt2`, `gpt2-medium`, `gpt2-large`) |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `2` | Batch size per device |
| `--lr` | `5e-5` | Learning rate |

Training saves the model to `output/model/` when complete.

---

## Running the Web App

Start the Flask server:

```bash
python app.py
```

Or with custom options:

```bash
python app.py --model output/model --port 5000 --debug
```

Open your browser at **http://localhost:5000** to use the chat interface.

The `/chat` endpoint accepts POST requests:

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is deep learning?"}'
```

Response:

```json
{
  "question": "What is deep learning?",
  "response": "Deep learning is a branch of machine learning that uses artificial neural networks..."
}
```

---

## Console Inference

For quick testing without the web app:

```bash
# Interactive mode
python inference.py

# Single question
python inference.py --question "What is a neural network?"
```

---

## Sample Output

```
You: What is gradient descent?

GPT: Gradient descent is an optimization algorithm used to minimize the
     loss function by iteratively updating model parameters in the
     direction of the negative gradient. Variants include stochastic
     gradient descent (SGD), mini-batch gradient descent, Adam, and RMSprop.
```

> **Screenshot placeholder** — Add a screenshot of the web UI here after running `app.py`.

---

## Customizing for Your Domain

1. Replace `data/train.txt` with your domain's Q&A pairs (any topic: legal, medical, finance, customer support, etc.)
2. Retrain with `python train.py`
3. The model will specialize in answering questions from your domain

For better quality, provide 50–200+ high-quality Q&A pairs.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language Model | GPT-2 (Hugging Face Transformers) |
| Fine-tuning | Hugging Face `Trainer` API |
| Web Framework | Flask |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Data Format | Plain text Q&A pairs |

---

## License

MIT License

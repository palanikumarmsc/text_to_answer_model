# 🧠 text_to_answer_model

A simple Question Answering (QA) system trained on synthetic story-based datasets. This project demonstrates how to generate custom QA datasets, train a BERT-based model using Hugging Face Transformers, and use it to answer questions from short narratives.

---

## 🚀 Project Overview

This project helps you build a small language model that:
- Reads a short story (context)
- Answers a question based on that story

You will:
- Generate synthetic QA data
- Train a BERT QA model using Hugging Face's `transformers` and `datasets`
- Evaluate it by asking questions from unseen contexts

---

## 🛠️ Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
```

### `requirements.txt`:
```
transformers
datasets
evaluate
torch
```

---

## 📄 Files in the Repo

### ✅ `generate.py`
Generates synthetic QA data in [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) format using predefined short stories. It saves the data in `generated_qa.json`.

```bash
python generate.py
```

### ✅ `train.py`
Trains a BERT-based QA model (`bert-base-uncased`) using the generated data.

```bash
python train.py
```

This will:
- Tokenize the data
- Train the model for 2 epochs
- Save the model and tokenizer to `./bert-qa/`

### ✅ `test.py`
Tests the trained model on a new example context and question.

```bash
python test.py
```

Sample output:
```
Q: What did Sara bake?
A: a chocolate cake
```

---

## 📈 How It Works

1. **Data Generation**:
   - Predefined short stories are randomly chosen
   - For each story, a question-answer pair is generated
   - Data is formatted in SQuAD-style JSON

2. **Training**:
   - A pretrained BERT model is fine-tuned on the generated QA pairs
   - The model learns to predict the answer span in a given context

3. **Inference**:
   - Given a new story and question, the model extracts the most probable answer span

---

## 💡 Example Stories Used

- "John went to the market to buy vegetables..."
- "Alice was reading a mystery novel in the library..."
- "Tommy loves playing football..."
- "Sara baked a chocolate cake..."
- "David and Emma went to the zoo..."

---

## 📚 Future Improvements

- Expand the variety of story templates
- Evaluate the model on real-world data
- Add support for multiple QA pairs per story
- Integrate a web or chatbot interface

---

## 🧑‍💻 Author

**Palanikumar M**  
[GitHub Repository](https://github.com/palanikumarmsc/text_to_answer_model.git)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

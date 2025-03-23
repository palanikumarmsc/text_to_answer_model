from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch

# Load trained model and tokenizer
model_path = "./bert-qa"
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForQuestionAnswering.from_pretrained(model_path)

# Example story and question
context = "Sara baked a chocolate cake for her brother's birthday."
question = "What did Sara bake?"

# Encode inputs
inputs = tokenizer(question, context, return_tensors="pt")
input_ids = inputs["input_ids"].tolist()[0]

# Get model prediction
with torch.no_grad():
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

# Decode the answer
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
)

print(f"Q: {question}")
print(f"A: {answer}")

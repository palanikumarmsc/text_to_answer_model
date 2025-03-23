from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset

# Load SQuAD format data
dataset = load_dataset('json', data_files={'train': 'generated_qa.json'}, field='data')

# Flatten the data (for HuggingFace compatibility)
def flatten(example):
    return {
        'title': example['title'],
        'context': example['paragraphs'][0]['context'],
        'question': example['paragraphs'][0]['qas'][0]['question'],
        'answers': example['paragraphs'][0]['qas'][0]['answers']
    }

dataset = dataset.map(flatten)

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Tokenize data
def preprocess(example):
    return tokenizer(
        example['question'],
        example['context'],
        truncation=True,
        padding='max_length',
        max_length=512,
        return_offsets_mapping=True
    )

tokenized = dataset['train'].map(preprocess, batched=True)

# Add start and end positions
def add_labels(example):
    answer = example['answers'][0]  # Access first answer dict
    start_char = answer['answer_start']
    end_char = start_char + len(answer['text'])

    offsets = example['offset_mapping']
    start_pos = end_pos = 0

    for i, (start, end) in enumerate(offsets):
        if start <= start_char < end:
            start_pos = i
        if start < end_char <= end:
            end_pos = i
            break

    example['start_positions'] = start_pos
    example['end_positions'] = end_pos
    return example


tokenized = tokenized.map(add_labels)

# Remove unnecessary columns
tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])

# Training
training_args = TrainingArguments(
    output_dir='./bert-qa',
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

trainer.train()

# Save the model
model.save_pretrained("./bert-qa")
tokenizer.save_pretrained("./bert-qa")

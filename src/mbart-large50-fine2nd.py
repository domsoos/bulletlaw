import os
import time
import json
import torch

from sklearn.model_selection import train_test_split
from transformers import AdamW, MBartForConditionalGeneration, MBartTokenizer
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

import torch.nn as nn

from rouge import Rouge
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score

nltk.download('punkt')

num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

print("Is CUDA available:", torch.cuda.is_available())
print("Number of GPU(s):", torch.cuda.device_count())

if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(current_device)
    print(f"Device Name: {gpu_properties.name}")
    print(f"Number of Streaming Multiprocessors (SMs): {gpu_properties.multi_processor_count}")
else:
    print("No GPU available.")


# Load data from JSON file
def load_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def split_text_into_chunks(text, max_length):
    """
    Splits a given text into chunks of a maximum specified length.
    Args:
    - text (str): The text to be split.
    - max_length (int): The maximum length of each chunk.

    Returns:
    - list of str: List of text chunks.
    """
    chunks = []
    while text:
        split_index = text.rfind(' ', 0, max_length)
        if split_index == -1:  # No space found, force split
            split_index = max_length
        chunk = text[:split_index].strip()
        chunks.append(chunk)
        text = text[split_index:].strip()
    return chunks

def prepare_data(sample, max_length=1024):
    """
    Extracts 'input' text from a sample, splits it into chunks, and pairs each chunk with the 'output>
    Args:
    - sample (dict): A sample from the data containing 'title', 'input', and 'output'.
    - max_length (int): The maximum length of each chunk.

    Returns:
    - list of tuples: Each tuple contains a chunk of text and the corresponding output.
    """
    input_text = sample["input"]
    output_text = sample["output"]
    chunks = split_text_into_chunks(input_text, max_length)

    # Pair each chunk with the output
    processed_samples = [(chunk, output_text) for chunk in chunks]
    return processed_samples

# Split data into training, validation, and testing sets
def split_data(data, test_size=0.2, val_size=0.1):
    train_val, test = train_test_split(data, test_size=test_size, random_state=42)
    train_size = 1 - val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=train_size, random_state=42)
    return train, val, test



data = load_data("./ground_truth_multi.json")

# Process each sample
processed_data = []
for sample in data:
    processed_samples = prepare_data(sample)
    processed_data.append(processed_samples) 
print(f"number of samples in processed data: {len(processed_data)}")
# Define the split sizes
train_size = 0.8 # 16 samples
test_size = 0.1  # 2
val_size = 0.1   # 2

# First, split into training and non-training
train_data, non_train_data = train_test_split(processed_data, train_size=train_size, random_state=42, shuffle=True)

train_data = train_data*2

# Then split non-training data into validation and test
val_data, test_data = train_test_split(non_train_data, test_size=test_size / (test_size + val_size), random_state=42, shuffle=True)
print(len(train_data[0][0][0]))
print(type(train_data[0][0]))


num_epochs = 8
batch_size = 4

#configuration = BartConfig()

#print(configuration)

# Check if GPU is available, and if so, move the model to the GPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.

# Load pre-trained model and tokenizer
model_name = "facebook/mbart-large-50"
tokenizer = MBartTokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
model.cuda()
model = nn.DataParallel(model, device_ids=[0,1,2,3])
model = model.to(device)
print(f"tokenizer + model loaded onto {device}")

# Move the data to the same device as the model
train_encodings = tokenizer([chunk for sample in train_data for chunk, _ in sample], max_length=1024, padding=True, truncation=True)
train_labels = tokenizer([summary for sample in train_data for _, summary in sample], max_length=128, padding=True, truncation=True)
input_ids = torch.tensor(train_encodings['input_ids']).to(device)
attention_masks = torch.tensor(train_encodings['attention_mask']).to(device)
labels = torch.tensor(train_labels['input_ids']).to(device)

# Validation data
val_encodings = tokenizer([chunk for sample in val_data for chunk, _ in sample], max_length=1024, padding=True, truncation=True)
val_labels = tokenizer([summary for sample in val_data for _, summary in sample], max_length=128, padding=True, truncation=True)
val_input_ids = torch.tensor(val_encodings['input_ids']).to(device)
val_attention_masks = torch.tensor(val_encodings['attention_mask']).to(device)
val_labels = torch.tensor(val_labels['input_ids']).to(device)


test_encodings = tokenizer([chunk for sample in test_data for chunk, _ in sample], max_length=1024, padding=True, truncation=True)
test_labels = tokenizer([summary for sample in test_data for _, summary in sample], max_length=128, padding=True, truncation=True)
test_input_ids = torch.tensor(test_encodings['input_ids']).to(device)
test_attention_masks = torch.tensor(test_encodings['attention_mask']).to(device)
test_labels = torch.tensor(test_labels['input_ids']).to(device)

# Create a DataLoader for testing
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define a loss function and optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

from torch.utils.data import DataLoader, TensorDataset

# Create a TensorDataset for training and validation
train_dataset = TensorDataset(input_ids, attention_masks, labels)
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
scalar_loss = 0
# Training loop
#num_epochs = 1
elapsed = 0

def format_time(elapsed):
    return str(int(elapsed // 3600)) + ":" + str(int((elapsed % 3600) // 60)) + ":" + str(int((elapsed % 3600) % 60))

for epoch in range(num_epochs):
    t0 = time.time()
    model.train()
    total_train_loss = 0.0
    correct_train_predictions = 0
    total_train_samples = 0
    optimizer.zero_grad()
    for batch in train_loader:
        if epoch % 50 == 0 and not epoch == 0:
            elapsed = format_time(time.time() - t0)
        # Unpack the batch
        input_ids, attention_mask, labels = batch

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        #loss = outputs.loss

        loss = outputs[0]
        total_train_loss += torch.sum(loss).item()
        scalar_loss = torch.sum(loss)

        # Backward pass
        scalar_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()


        # Calculate training accuracy
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        correct_train_predictions += torch.sum(predicted_ids == labels).item()
        total_train_samples += labels.size(0)

        #total_train_loss += loss.item()

    # Calculate training accuracy and loss
    train_accuracy = correct_train_predictions / total_train_samples
    train_loss = total_train_loss / len(train_loader)

    print(f"training epoch #{epoch} took: {format_time(time.time() - t0)} | loss = {train_loss:.4f} | accuracy = {train_accuracy:.4f}")
    # Validation loop
    model.eval()
    total_val_loss = 0.0
    correct_val_predictions = 0
    total_val_samples = 0

    for batch in val_loader:
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs[0]

        # Calculate validation accuracy
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        correct_val_predictions += torch.sum(predicted_ids == labels).item()
        total_val_samples += labels.size(0)

        total_val_loss += torch.sum(loss)

    # Calculate validation accuracy and loss
    val_accuracy = correct_val_predictions / total_val_samples
    val_loss = total_val_loss / len(val_loader)

    print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

# Save the fine-tuned model
# Save the original model wrapped inside DataParallel
if isinstance(model, nn.DataParallel):
    model.module.save_pretrained('./model2')
else:
    model.save_pretrained('./model2')
#model.save_pretrained('./model2')


# Testing loop
model.eval()
total_test_loss = 0.0
correct_test_predictions = 0
total_test_samples = 0
all_predictions = []  # To collect all predictions
all_labels = []  # To collect all ground truth labels

for batch in test_loader:
    input_ids, attention_mask, labels = batch
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    loss = outputs[0]

    # Calculate testing accuracy
    logits = outputs.logits
    predicted_ids = torch.argmax(logits, dim=-1)
    correct_test_predictions += torch.sum(predicted_ids == labels).item()
    total_test_samples += labels.size(0)

    total_test_loss += torch.sum(loss)

    # Collect predictions and ground truth labels
    all_predictions.extend(predicted_ids.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

# Calculate testing accuracy and loss
test_accuracy = correct_test_predictions / total_test_samples
test_loss = total_test_loss / len(test_loader)

print(f"Testing Loss: {test_loss:.4f} | Testing Accuracy: {test_accuracy:.4f}")

# Testing loop with ROUGE, BLEU, and BERTScore evaluation
model.eval()
generated_texts = []
reference_texts = []

for batch in test_loader:
    input_ids, attention_mask, labels = batch
    with torch.no_grad():
# When calling the generate method
        if isinstance(model, nn.DataParallel):
             generated_outputs = model.module.generate(input_ids=input_ids, attention_mask=attention_mask)
        else:
             generated_outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)

        #outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
    
    # Decode the generated and reference texts
    generated_batch_texts = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_outputs]
    reference_batch_texts = [tokenizer.decode(l, skip_special_tokens=True, clean_up_tokenization_spaces=True) for l in labels]

    generated_texts.extend(generated_batch_texts)
    reference_texts.extend(reference_batch_texts)

with open("generated_text.txt", "w") as file:
    for text in generated_texts:
        file.write(text)
        file.write("\n\n")

# Compute ROUGE scores
rouge = Rouge()
rouge_scores = rouge.get_scores(generated_texts, reference_texts, avg=True)

# Compute BLEU scores
smoothie = SmoothingFunction().method4
bleu_scores = [sentence_bleu([ref.split()], gen.split(), smoothing_function=smoothie) for gen, ref in zip(generated_texts, reference_texts)]


# Compute BERTScore
P, R, F1 = score(generated_texts, reference_texts, lang="en", rescale_with_baseline=True)

print("ROUGE Scores:", rouge_scores)
print("Average BLEU Score:", sum(bleu_scores) / len(bleu_scores))
print("BERTScore:", {"Precision": P.mean().item(), "Recall": R.mean().item(), "F1": F1.mean().item()})

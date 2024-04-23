from transformers import BertTokenizer
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import tqdm
def encode_text(texts, tokenizer):
    """ Encodes a list of texts into BERT's format. """
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,                      # Text to encode
            add_special_tokens=True,   # Add '[CLS]' and '[SEP]'
            max_length=512,            # Pad & truncate all sentences
            padding='max_length',
            return_attention_mask=True, # Construct attention masks
            truncation=True,
            return_tensors='pt'        # Return PyTorch tensors
        )
        input_ids.append(encoded['input_ids'][0])
        attention_masks.append(encoded['attention_mask'][0])

    # Stack the tensors to create a single tensor for all input ids and masks
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)

    return input_ids, attention_masks

def load_data(csv_file, tokenizer):
    """ Loads data from CSV and encodes it using the BERT tokenizer. """
    data = pd.read_csv(csv_file, low_memory=False)
    print(f"data['id'].dtype = {data['id'].dtype}")  # Check the data type of 'id' column

    # Ensure 'id' column is integer
    data['id'] = pd.to_numeric(data['id'], errors='coerce')  # Convert to numeric, set errors to 'coerce' to handle non-numeric values
    data = data.dropna(subset=['id'])  # Drop rows where 'id' is NaN after conversion
    data['id'] = data['id'].astype(int)  # Convert to int

    #dealing with overview
    data = data.dropna(subset=['overview'])
    data = data[data['overview'].str.strip() != '']
    # Encoding text data using the encode_text function
    input_ids, attention_masks = encode_text(data['overview'].tolist(), tokenizer)

    # Optionally, convert IDs to tensor if you want to use them as labels or for indexing
    ids = torch.tensor(data['id'].values, dtype=torch.int64)  # Explicitly specify dtype as int64

    return input_ids, attention_masks, ids



class MovieClassifier(nn.Module):
    def __init__(self, num_movies):
        super(MovieClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_movies)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

# Assuming the load_data function returns all data needed for splitting
# def load_data_with_split(csv_file, tokenizer, test_size=0.1):
#     input_ids, attention_masks, movie_ids = load_data(csv_file, tokenizer)

#     return dataset = TensorDataset(input_ids, attention_masks, ids)

def evaluate_model(model, eval_loader, device):
    model.eval()
    correct = 0
    total = 0
    eval_iterator = tqdm(eval_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in eval_iterator:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

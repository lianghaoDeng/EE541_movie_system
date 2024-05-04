from transformers import BertTokenizer
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import json
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
    data = pd.read_csv(csv_file, low_memory=False)
    # data = data[0:80] #for exmaple testing 
    data = data.dropna(subset=['overview', 'genres'])
    data['genres'] = data['genres'].apply(lambda x: json.loads(x.replace("'", "\"")))

    # Encode the textual data into BERT's format
    input_ids, attention_masks = encode_text(data['overview'].tolist(), tokenizer)

    # Extract genre ids and prepare them for multi-label classification
    genre_lists = [list(set(genre['id'] for genre in genres)) for genres in data['genres']]
    mlb = MultiLabelBinarizer()
    encoded_genres = mlb.fit_transform(genre_lists)

    return input_ids, attention_masks, torch.tensor(encoded_genres, dtype=torch.float32)


class MovieClassifier(nn.Module):
    def __init__(self, num_movies):
        super(MovieClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)  # Add a Dropout layer
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_movies)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped_out = self.dropout(pooled_output)
        logits = self.classifier(dropped_out)
        return logits

# Assuming the load_data function returns all data needed for splitting
# def load_data_with_split(csv_file, tokenizer, test_size=0.1):
#     input_ids, attention_masks, movie_ids = load_data(csv_file, tokenizer)

#     return dataset = TensorDataset(input_ids, attention_masks, ids)

# def evaluate_model(model, eval_loader, device):
#     model.eval()
#     correct = 0
#     total = 0
#     eval_iterator = tqdm(eval_loader, desc="Evaluating", leave=False)

#     with torch.no_grad():
#         for batch in eval_iterator:
#             input_ids, attention_mask, labels = [b.to(device) for b in batch]
#             outputs = model(input_ids, attention_mask)
#             predictions = torch.sigmoid(outputs) > 0.5  # Convert logits to binary predictions
#             correct += (predictions == labels).all(dim=1).sum().item()  # Correct only if all labels match
#             total += labels.size(0)


#     accuracy = 100 * correct / total
#     return accuracy

# def evaluate_model(model, data_loader, device):
#     model.eval()
#     total_samples = 0
#     total_correct = 0

#     with torch.no_grad():
#         for inputs, labels in data_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#             predictions = torch.sigmoid(outputs) > 0.5  # Convert logits to binary predictions
#             total_correct += (predictions == labels).all(dim=1).sum().item()  # Correct only if all labels match
#             total_samples += labels.size(0)

#     accuracy = total_correct / total_samples
#     return accuracy * 100  # convert fraction to percentage

def evaluate_model(model, eval_loader, device, top_k=3):

    total_samples = 0
    top_k_matches = 0
    eval_iterator = tqdm(eval_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in eval_iterator:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            top_k_probs, top_k_preds = torch.topk(probs, top_k, dim=1)

            for i in range(labels.size(0)):
                actual_labels = labels[i].bool()
                predicted_labels = torch.zeros_like(labels[i], dtype=torch.bool)
                predicted_labels.scatter_(0, top_k_preds[i], 1)
                
                if (predicted_labels & actual_labels).sum() > 0:  # If there's any overlap
                    top_k_matches += 1

            total_samples += labels.size(0)

    accuracy = 100 * top_k_matches / total_samples
    return accuracy
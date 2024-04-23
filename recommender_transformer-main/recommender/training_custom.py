from transformers import BertTokenizer
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import BertModel
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from train_utils import MovieClassifier, encode_text, load_data, load_data_with_split, evaluate_model
from sklearn.model_selection import train_test_split



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

csv_file_path = '../../data/kaggle_movie/movies_metadata.csv'
from sklearn.model_selection import train_test_split



(train_ids, train_masks, train_labels), (val_ids, val_masks, val_labels) = load_data_with_split(csv_file_path, tokenizer, test_size=0.2)

# Encode labels using LabelEncoder and determine number of classes
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
val_labels = label_encoder.transform(val_labels)

num_classes = len(np.unique(train_labels)) #compute the numbers of classes 

train_dataset = TensorDataset(train_ids, train_masks, torch.tensor(train_labels))
val_dataset = TensorDataset(val_ids, val_masks, torch.tensor(val_labels))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


dataset = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"This is number of classes inside the movie dataset")
model = MovieClassifier(num_movies=num_classes) 
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

epochs = 10
for epoch in range(epochs):
    for input_ids, attention_mask, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    valid_acc = evaluate_model(model, val_loader)
    print(f"epoch = {epoch}, valid accuracy = {valid_acc}")
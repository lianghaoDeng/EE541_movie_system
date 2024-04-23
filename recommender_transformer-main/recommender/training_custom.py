from transformers import BertTokenizer
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import BertModel
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split, TensorDataset
from train_utils import MovieClassifier, encode_text, load_data, load_data_with_split, evaluate_model
from sklearn.model_selection import train_test_split



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

csv_file_path = '../../data/kaggle_movie/movies_metadata.csv'



input_ids, attention_masks, ids = load_data(csv_file=csv_file_path, tokenizer=tokenizer)


# Encode labels using LabelEncoder and determine number of classes
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(ids) 

dataset = TensorDataset(input_ids, attention_masks, encoded_labels)


train_size = int(0.7 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)


num_classes = len(np.unique(encoded_labels)) #compute the numbers of classes 


print(f"Data loading done! This is number of classes = {num_classes} inside the movie dataset")
model = MovieClassifier(num_movies=num_classes) 
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

epochs = 10
for epoch in range(epochs):
    for input_ids, attention_mask, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    valid_acc = evaluate_model(model, eval_loader)
    print(f"epoch = {epoch}, valid accuracy = {valid_acc}")
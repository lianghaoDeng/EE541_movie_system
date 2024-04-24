from transformers import BertTokenizer
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import BertModel
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split, TensorDataset
from train_utils import MovieClassifier, encode_text, load_data, evaluate_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import os
import glob



def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids, attention_masks, ids = load_data(csv_file=args.csv_file_path, tokenizer=tokenizer)

    # Encode labels using LabelEncoder and determine number of classes
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(ids.numpy()) 
    print(f"Type of input_ids: {type(input_ids)}, shape: {input_ids.shape}")
    print(f"Type of attention_masks: {type(attention_masks)}, shape: {attention_masks.shape}")
    print(f"Type of encoded_labels: {type(encoded_labels)}, shape: {torch.tensor(encoded_labels, dtype=torch.long).shape}")
    if not isinstance(encoded_labels, torch.Tensor):
        encoded_labels = torch.tensor(encoded_labels, dtype=torch.long)
    else:
        encoded_labels = encoded_labels.type(torch.long)

    print(f"Type of encoded_labels after conversion/check: {type(encoded_labels)}")
    dataset = TensorDataset(input_ids, attention_masks, encoded_labels)


    train_size = int(0.7 * len(dataset))
    eval_size = len(dataset) - train_size

    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)


    num_classes = len(np.unique(ids)) #compute the numbers of classes 


    
    print(f"Data loading done! This is number of classes = {num_classes} inside the movie dataset")
    
    if args.load_best_model:
        model_files = glob.glob('customBert_*.pt')
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)  # Load the latest best model
            model = MovieClassifier(num_movies=num_classes).to(device)
            model.load_state_dict(torch.load(latest_model))
            print(f"Loaded best model from {latest_model}")
        else:
            print("No best model found, initializing from pretrained model instead.")
            model = MovieClassifier(num_movies=num_classes).to(device)
    else:
        model = MovieClassifier(num_movies=num_classes).to(device)
    
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    best_valid_acc = 0.0 
    epochs = args.epochs
    for epoch in range(epochs):

        model.train()
        train_iterator = tqdm(train_loader, desc="Training", leave=False)
        for batch in train_iterator:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            train_iterator.set_description(f"Training (loss={loss.item():.4f})")

        model.eval()
        valid_acc = evaluate_model(model, train_loader, device=device)
        print(f"Epoch {epoch}: Validation Accuracy = {valid_acc}%")
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            model_path = f"./customBert_{valid_acc:.3f}.pt"  # Format the accuracy to 2 decimal places
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a movie classifier model.")
    parser.add_argument('--csv_file_path', type=str, default='../../data/kaggle_movie/movies_metadata.csv',
                        help='Path to the CSV file containing the movie data.')
    parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs')
    parser.add_argument('--load_best_model', action='store_true',
                    help='Load the best previously trained model if set, otherwise load a pretrained model.')

    args = parser.parse_args()
    main(args)
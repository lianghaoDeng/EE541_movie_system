from transformers import BertTokenizer
import torch
from torch import nn
from transformers import BertModel
from torch.utils.data import DataLoader, TensorDataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_data(csv_file):
    """ Loads data from CSV and encodes it using the BERT tokenizer. """
    data = pd.read_csv(csv_file)
    
    # Encoding text data using the encode_text function
    input_ids, attention_masks = encode_text(data['overview'].tolist())

    # Optionally, convert IDs to tensor if you want to use them as labels or for indexing
    ids = torch.tensor(data['id'].values)

    return input_ids, attention_masks, ids

# Path to your CSV file
csv_file_path = '../data/kaggle_movie/movies_metadata.csv'
input_ids, attention_masks, movie_ids = load_data(csv_file_path)

# Function to encode the text data
def encode_text(data):
    input_ids = []
    attention_masks = []
    
    for item in data:
        encoded = tokenizer.encode_plus(
            item['overview'],                  # Text to encode
            add_special_tokens=True,           # Add '[CLS]' and '[SEP]'
            max_length=512,                    # Pad & truncate all sentences
            padding='max_length',
            return_attention_mask=True,        # Construct attention masks
            return_tensors='pt',               # Return PyTorch tensors
            truncation=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    # Stack the tensors to create a single tensor for all input ids and masks
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    return input_ids, attention_masks

input_ids, attention_masks = encode_text(data)

# Convert labels to tensor
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform([item['id'] for item in data])  # Convert movie IDs to integer labels
labels = torch.tensor(labels)




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
    



dataset = TensorDataset(encoded_keywords, attention_masks, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = MovieClassifier(num_movies=1000)  # Assuming 1000 different movies
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

for input_ids, attention_mask, labels in dataloader:
    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    optimizer.step()


model.eval()
keywords = ["adventure", "hero", "action"]
input_ids, attention_mask = encode_keywords(keywords)
logits = model(input_ids, attention_mask)
predicted_movie_id = logits.argmax(-1)
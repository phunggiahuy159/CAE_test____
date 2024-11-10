import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer

# Define aspect and sentiment mappings
aspect2idx = {
    'CAMERA': 0, 'FEATURES': 1, 'BATTERY': 2, 'PERFORMANCE': 3,
    'DESIGN': 4, 'GENERAL': 5, 'PRICE': 6, 'SCREEN': 7, 'SER&ACC': 8, 'STORAGE': 9
}
sentiment2idx = {
    'Positive': 2, 'Neutral': 1, 'Negative': 0
}
num_aspect = len(aspect2idx)

# Convert label cell to tensor
def convert_label(cell):
    return torch.tensor([float(x) for x in cell.strip('[]').split()])

# Load train data
train = pd.read_csv("D:/code/intro_ai_ABSA/Train_preprocessed_with_-1.csv")
sentences_train = list(train['comment'])
labels_train = list(train['label'].apply(convert_label))

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')

# Define dataset
class CustomTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        dic = {key: val.squeeze(0) for key, val in encoding.items()}
        dic['labels'] = labels
        return dic

# Create dataset and dataloader
train_dataset = CustomTextDataset(sentences_train, labels_train, tokenizer)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=128)

# Define the LSTM-based model
class Cae(nn.Module):
    def __init__(self, word_embedder, categories, polarities):
        super().__init__()
        self.word_embedder = word_embedder
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss(ignore_index=-1)

        embed_dim = word_embedder.embedding_dim
        self.lstm = nn.LSTM(embed_dim, embed_dim // 2, batch_first=True, bidirectional=True)
        self.dropout_after_lstm = nn.Dropout(0.25)

        self.category_fcs = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1)) for _ in range(self.category_num)])
        self.sentiment_fc = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, self.polarity_num)) for _ in range(self.category_num)])

    def forward(self, tokens, labels, mask):
        word_embeddings = self.word_embedder(tokens)
        
        # Pass through LSTM
        lstm_result, _ = self.lstm(word_embeddings)
        lstm_result = self.dropout_after_lstm(lstm_result)

        # Pool the LSTM output (mean pooling) for each token's hidden states
        pooled_output = lstm_result.mean(dim=1)

        final_category_outputs = []
        final_sentiment_outputs = []
        
        for i in range(self.category_num):
            # Category and sentiment predictions
            category_output = self.category_fcs[i](pooled_output)
            sentiment_output = self.sentiment_fc[i](pooled_output)
            final_category_outputs.append(category_output)
            final_sentiment_outputs.append(sentiment_output)

        loss = 0
        if labels is not None:
            category_labels = labels[:, :self.category_num]
            polarity_labels = labels[:, self.category_num:]

            for i in range(self.category_num):
                category_mask = (category_labels[:, i] != -1)  # Mask out ignored labels
                sentiment_mask = (polarity_labels[:, i] != -1)

                if category_mask.any():  # Only calculate if there are valid labels
                    category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(-1)[category_mask], category_labels[:, i][category_mask])
                    loss += category_temp_loss

                if sentiment_mask.any():  # Only calculate if there are valid labels
                    sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i][sentiment_mask], polarity_labels[:, i][sentiment_mask].long())
                    loss += sentiment_temp_loss

        output = {
            'pred_category': [torch.sigmoid(e) for e in final_category_outputs],
            'pred_sentiment': [torch.softmax(e, dim=-1) for e in final_sentiment_outputs]
        }
        
        return output, loss

# Load embedding matrix and initialize model
w2v = 'D:/code/intro_ai_ABSA/CAE____/W2V_150.txt'
embedding_dim = 150
word_to_vec = {}
with open(w2v, 'r', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        word_to_vec[word] = vector

vocab = tokenizer.get_vocab()
vocab_size = len(vocab)
E = np.zeros((vocab_size, embedding_dim))
for word, idx in vocab.items():
    E[idx] = word_to_vec.get(word, np.random.normal(scale=0.6, size=(embedding_dim,)))

embedding_matrix = torch.tensor(E, dtype=torch.float32)
embedding_layer = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

categories = aspect2idx.keys()
polarities = sentiment2idx.keys()
model = Cae(embedding_layer, categories, polarities)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Training loop
epochs = 30
l2_lambda = 0.01
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        output, loss = model(batch['input_ids'], batch['labels'], batch['attention_mask'])
        l2_reg = torch.tensor(0., requires_grad=True)  # Initialize L2 regularization term
        for param in model.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)  # Add squared norm of each parameter

        # Add L2 regularization term to loss
        loss = loss + l2_lambda * l2_reg
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1} - Loss: {total_loss / len(train_dataloader)}")
torch.save(model.state_dict(), "test.pth")

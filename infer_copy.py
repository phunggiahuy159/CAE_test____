import torch
import torch.nn as nn
import numpy as np 
import pandas as pd
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
# Define aspect and sentiment mappings
num_aspect=10
aspect2idx = {
    'CAMERA': 0, 'FEATURES': 1, 'BATTERY': 2, 'PERFORMANCE': 3,
    'DESIGN': 4, 'GENERAL': 5, 'PRICE': 6, 'SCREEN': 7, 'SER&ACC': 8, 'STORAGE': 9
}
sentiment2idx = {
    'Positive': 2, 'Neutral': 1, 'Negative': 0
}
embedding_dim=300

vocab=tokenizer.get_vocab()
vocab_size=len(vocab)

embedding_layer=nn.Embedding(vocab_size,embedding_dim)
vocab=tokenizer.get_vocab()
vocab_size=len(vocab)
categories=aspect2idx.keys()
polarities=sentiment2idx.keys()

class AttentionInHtt(nn.Module):
    def __init__(self, in_features, out_features, bias=True, softmax=True):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias)
        self.uw = nn.Linear(out_features, 1, bias=False)
        self.softmax = softmax

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        u = self.W(h)  # (batch_size, seq_len, out_features)
        u = torch.tanh(u)
        similarities = self.uw(u)  # (batch_size, seq_len, 1)
        similarities = similarities.squeeze(dim=-1)  # (batch_size, seq_len)

        # Mask the similarities
        similarities = similarities.masked_fill(~mask.bool(), -float('inf'))

        if self.softmax:
            alpha = torch.softmax(similarities, dim=-1)
            return alpha
        else:
            return similarities

def element_wise_mul(input1, input2, return_not_sum_result=False):
        output = input1 * input2.unsqueeze(2)  # Ensure correct broadcasting
        result = output.sum(dim=1)
        if return_not_sum_result:
            return result, output
        else:
            return result
class Cae(nn.Module):
    def __init__(self, word_embedder, categories, polarities):
        super().__init__()
        self.word_embedder = word_embedder
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss(ignore_index=-1)

        embed_dim = word_embedder.embedding_dim
        self.embedding_layer_fc = nn.Linear(embed_dim, embed_dim)
        self.embedding_layer_aspect_attentions = nn.ModuleList([AttentionInHtt(embed_dim, embed_dim) for _ in range(self.category_num)])
        self.lstm_layer_aspect_attentions = nn.ModuleList([AttentionInHtt(embed_dim, embed_dim) for _ in range(self.category_num)])

        self.lstm = nn.LSTM(embed_dim, embed_dim // 2, batch_first=True, bidirectional=True)
        self.dropout_after_embedding = nn.Dropout(0.5)
        self.dropout_after_lstm = nn.Dropout(0.5)

        self.category_fcs = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim * 2, 32), nn.ReLU(), nn.Linear(32, 1)) for _ in range(self.category_num)])
        # self.sentiment_fc = nn.Sequential(nn.Linear(embed_dim * 2, 32), nn.ReLU(), nn.Linear(32, self.polarity_num))
        self.sentiment_fc = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim * 2, 32), nn.ReLU(), nn.Linear(32, self.polarity_num)) for _ in range(self.category_num)])

    def forward(self, tokens, mask):
        word_embeddings = self.word_embedder(tokens)
        word_embeddings = self.dropout_after_embedding(word_embeddings)

        embeddings = word_embeddings
        embedding_layer_category_outputs = []
        embedding_layer_sentiment_outputs = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(embeddings, mask)

            category_output = element_wise_mul(embeddings, alpha)
            embedding_layer_category_outputs.append(category_output)

            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, embeddings.transpose(1, 2)).squeeze(1)
            sentiment_alpha = softmax(sentiment_alpha, dim=-1)
            sentiment_output = torch.matmul(sentiment_alpha.unsqueeze(1), word_embeddings).squeeze(1)
            embedding_layer_sentiment_outputs.append(sentiment_output)

        lstm_result, _ = self.lstm(word_embeddings)
        lstm_result = self.dropout_after_lstm(lstm_result)
        
        lstm_layer_category_outputs = []
        lstm_layer_sentiment_outputs = []
        for i in range(self.category_num):
            lstm_layer_aspect_attention = self.lstm_layer_aspect_attentions[i]
            alpha = lstm_layer_aspect_attention(lstm_result, mask)
            category_output = element_wise_mul(lstm_result, alpha)
            lstm_layer_category_outputs.append(category_output)

            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, lstm_result.transpose(1, 2)).squeeze(1)
            sentiment_alpha = softmax(sentiment_alpha, dim=-1)
            sentiment_output = torch.matmul(sentiment_alpha.unsqueeze(1), lstm_result).squeeze(1)
            lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc_detect = self.category_fcs[i]
            fc_clf = self.sentiment_fc[i]
            category_output = torch.cat([embedding_layer_category_outputs[i], lstm_layer_category_outputs[i]], dim=-1)
            final_category_output = fc_detect(category_output)
            final_category_outputs.append(final_category_output)

            sentiment_output = torch.cat([embedding_layer_sentiment_outputs[i], lstm_layer_sentiment_outputs[i]], dim=-1)
            final_sentiment_output = fc_clf(sentiment_output)
            final_sentiment_outputs.append(final_sentiment_output)

        output = {
            'pred_category': [torch.sigmoid(e) for e in final_category_outputs],
            'pred_sentiment': [torch.softmax(e, dim=-1) for e in final_sentiment_outputs]
        }
        
        return output


# Load model and set it to evaluation mode
model = Cae(embedding_layer, categories, polarities)
model.load_state_dict(torch.load(r"D:\code\intro_ai_ABSA\300_dim\CAE_300_50epoch.pth"))
model.eval()


def infer_batch(input_ids, attention_mask, threshold=0.65):
    # Set the model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Forward pass through the model
        output = model(input_ids, mask=attention_mask)
    
    # Extract category and sentiment predictions
    pred_categories = [torch.sigmoid(e) for e in output['pred_category']]
    pred_sentiments = [torch.argmax(e, dim=-1) for e in output['pred_sentiment']]

    final_categories = []
    final_sentiments = []
    
    for i in range(len(pred_categories)):
        batch_category = []
        batch_sentiment = []
        for j, category_score in enumerate(pred_categories[i]):
            # Apply threshold for aspect detection
            if category_score >= threshold:
                batch_category.append(1)  # Aspect detected
                batch_sentiment.append(pred_sentiments[i][j].item())
            else:
                batch_category.append(0)  # Aspect not detected
                batch_sentiment.append(-1)  # Set sentiment to -1 for undetected aspect
        final_categories.append(batch_category)
        final_sentiments.append(batch_sentiment)

    return {
        "Detected Aspects": final_categories,
        "Aspect Sentiments": final_sentiments
    }
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np



def calculate_macro_metrics(all_pred_categories, all_pred_sentiments, all_true_labels, num_aspect):
    """
    Calculate macro-averaged Precision, Recall, and F1-score for Aspect Detection and Sentiment Detection.

    Parameters:
    - all_pred_categories: List of predicted categories (Aspect Detection) for each instance
    - all_pred_sentiments: List of predicted sentiments (Sentiment Detection) for each instance
    - all_true_labels: List of true labels with aspect and sentiment information
    - num_aspect: The number of aspect labels (to split the labels correctly)

    Returns:
    - Dictionary with macro-averaged Precision, Recall, and F1-score for Aspect and Sentiment Detection
    """

    # Separate true labels into aspects and sentiments based on num_aspect
    true_acd = [label[:num_aspect] for label in all_true_labels]  # True Aspect Detection labels
    true_acsa = [label[num_aspect:] for label in all_true_labels]  # True Sentiment Detection labels

    # Flatten lists if needed (this step assumes true_acd and true_acsa are lists of lists)
    true_acd = [item for sublist in true_acd for item in sublist]
    true_acsa = [item for sublist in true_acsa for item in sublist]

    pred_acd = [item for sublist in all_pred_categories for item in sublist]
    pred_acsa = [item for sublist in all_pred_sentiments for item in sublist]

    # Calculate Precision, Recall, and F1-score for Aspect Detection
    acd_precision = precision_score(true_acd, pred_acd, average="macro", zero_division=0)
    acd_recall = recall_score(true_acd, pred_acd, average="macro", zero_division=0)
    acd_f1 = f1_score(true_acd, pred_acd, average="macro", zero_division=0)

    # Calculate Precision, Recall, and F1-score for Sentiment Detection
    acsa_precision = precision_score(true_acsa, pred_acsa, average="macro", zero_division=0)
    acsa_recall = recall_score(true_acsa, pred_acsa, average="macro", zero_division=0)
    acsa_f1 = f1_score(true_acsa, pred_acsa, average="macro", zero_division=0)

    return {
        "Aspect Detection": {
            "Precision": acd_precision * 100,
            "Recall": acd_recall * 100,
            "F1-score": acd_f1 * 100,
        },
        "Sentiment Detection": {
            "Precision": acsa_precision * 100,
            "Recall": acsa_recall * 100,
            "F1-score": acsa_f1 * 100,
        }
    }

# Run inference on the test dataset

def convert_label(cell):
    return torch.tensor([float(x) for x in cell.strip('[]').split()])


train = pd.read_csv("D:\code\intro_ai_ABSA\Train_final.csv")
sentences_test = list(train['comment'])
labels_test = list(train['label'].apply(convert_label))

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
test_dataset = CustomTextDataset(sentences_test, labels_test, tokenizer)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=128)


# Initialize lists to collect predictions and true labels for all batches
all_pred_categories = []
all_pred_sentiments = []
all_true_labels = []

# Loop through the test DataLoader to process the entire test set
for batch in test_dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    true_labels = batch['labels'].tolist()

    # Infer batch predictions
    predictions = infer_batch(input_ids=input_ids, attention_mask=attention_mask, threshold=0.7)

    # Append predictions and true labels to the cumulative lists
    all_pred_categories.extend(predictions["Detected Aspects"])
    all_pred_sentiments.extend(predictions["Aspect Sentiments"])
    all_true_labels.extend(true_labels)

# Calculate metrics on the entire dataset
metrics = calculate_macro_metrics(all_pred_categories, all_pred_sentiments, all_true_labels,num_aspect=10)

# Print the overall metrics
print("Metrics on the entire test dataset:", metrics)

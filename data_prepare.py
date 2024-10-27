import pandas as pd
from pyvi import ViTokenizer
import numpy as np 
import string 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import f1__ as f1
from tqdm import tqdm


aspect2idx={'CAMERA' : 0,
            'FEATURES' : 1,
            'BATTERY':2,
            'PERFORMANCE' : 3,
            'DESIGN' : 4,
            'GENERAL' : 5,
            'PRICE' : 6,
            'SCREEN' : 7,
            'SER&ACC' : 8,
            'STORAGE' : 9 
}   
sentiment2idx={'Positive':2,
               'Neutral':1,
               'Negative':0
}

num_aspect = 10
num_sentiment = 3

def convert_label(text):
    text = text.replace('{OTHERS};', '')
    all_aspect = text.split(';')
    all_aspect = [x.strip(r"{}") for x in all_aspect if x]  
    res = np.zeros(2 * num_aspect)

    for x in all_aspect:
        cate, sent = x.split('#')
        if cate in aspect2idx and sent in sentiment2idx:
            cate_value = aspect2idx[cate]
            sent_value = sentiment2idx[sent]
            res[cate_value] = 1
            res[cate_value + num_aspect] = sent_value
    
    return res
    


punc = string.punctuation
tokenizer = ViTokenizer.tokenize

train_df = pd.read_csv("D:\code\intro_ai_ABSA\CAE____\Train.csv")
# dev_df = pd.read_csv("Dev.csv")
test_df = pd.read_csv("D:\code\intro_ai_ABSA\CAE____\Test.csv")

def lowercase(df):
    df['comment'] = df['comment'].str.lower()
    return df

def remove_punc(text):
    return text.translate(str.maketrans('', '', punc))

def final_rmv_punc(df):
    df['comment'] = df['comment'].apply(remove_punc)
    return df

def remove_num(df):
    df['comment'] = df['comment'].replace(to_replace=r'\d', value='', regex=True)
    return df

def tokenize(df):
    df['comment'] = df['comment'].apply(tokenizer)
    return df

def preprocess(df):
    df.drop(['n_star', 'date_time'],axis=1, inplace = True)
    df = lowercase(df)
    df = final_rmv_punc(df)
    df = remove_num(df)
    df = tokenize(df)
    df['label'] = df['label'].apply(convert_label)
    return df

train = preprocess(train_df)
test = preprocess(test_df)

sentences_train=list(train['comment'])
sentences_test=list(test['comment'])
labels_train=list(train['label'])
labels_test=list(test['label'])

tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
class CustomTextDataset(Dataset):
    def __init__(self, texts,labels, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels=labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels=self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        dic={key: val.squeeze(0) for key, val in encoding.items()}
        dic['label']=labels 
        return dic

# Create dataset
train_dataset=CustomTextDataset(sentences_train,labels_train, tokenizer)
test_dataset=CustomTextDataset(sentences_test,labels_test,tokenizer)

# Data collator and dataloader
data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader=DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator)
test_dataloader=DataLoader(test_dataset, shuffle=False, batch_size=8, collate_fn=data_collator)

#model
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
        self.categories = categories
        self.polarities = polarities
        self.category_num = len(categories)
        self.polarity_num = len(polarities)
        self.category_loss = nn.BCEWithLogitsLoss()
        self.sentiment_loss = nn.CrossEntropyLoss()
        self._f1=f1.BinaryF1(0.5)

        # Get the embedding dimension directly from the word embedder
        embed_dim = word_embedder.embedding_dim  
        self.embedding_layer_fc = nn.Linear(embed_dim, embed_dim, bias=True)
        self.embedding_layer_aspect_attentions = nn.ModuleList([AttentionInHtt(embed_dim, embed_dim) for 
        _ in range(self.category_num)])
        self.lstm_layer_aspect_attentions = nn.ModuleList([AttentionInHtt(embed_dim, embed_dim) for _ in range(self.category_num)])

        # self.lstm = nn.LSTM(embed_dim, int(embed_dim / 2), batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(embed_dim, int(embed_dim / 2), batch_first=True, bidirectional=True)

        self.dropout_after_embedding = nn.Dropout(0.5)
        self.dropout_after_lstm = nn.Dropout(0.5)

        self.category_fcs = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim * 2, 16), nn.ReLU(), nn.Linear(16, 1)) for _ in range(self.category_num)])
        self.sentiment_fc = nn.Sequential(nn.Linear(embed_dim * 2, 16), nn.ReLU(), nn.Linear(16, self.polarity_num))

    def forward(self, tokens, label, polarity_mask, mask):
        mask = (tokens != 0).type(torch.FloatTensor)  
        
        word_embeddings = self.word_embedder(tokens)
        word_embeddings = self.dropout_after_embedding(word_embeddings)

        embeddings = word_embeddings
        embedding_layer_category_outputs = []
        embedding_layer_sentiment_outputs = []
        for i in range(self.category_num):
            embedding_layer_aspect_attention = self.embedding_layer_aspect_attentions[i]
            alpha = embedding_layer_aspect_attention(embeddings, mask)

            category_output = element_wise_mul(embeddings, alpha, return_not_sum_result=False)
            embedding_layer_category_outputs.append(category_output)

            # sentiment
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
            category_output = element_wise_mul(lstm_result, alpha, return_not_sum_result=False)
            lstm_layer_category_outputs.append(category_output)

            # sentiment
            category_output = category_output.unsqueeze(1)
            sentiment_alpha = torch.matmul(category_output, lstm_result.transpose(1, 2))
            sentiment_alpha = sentiment_alpha.squeeze(1)
            sentiment_alpha = softmax(sentiment_alpha, dim=-1)
            sentiment_alpha = sentiment_alpha.unsqueeze(1)
            sentiment_output = torch.matmul(sentiment_alpha, lstm_result).squeeze(1)  # batch_size x 2*hidden_dim
            lstm_layer_sentiment_outputs.append(sentiment_output)

        final_category_outputs = []
        final_sentiment_outputs = []
        for i in range(self.category_num):
            fc = self.category_fcs[i]
            category_output = torch.cat([embedding_layer_category_outputs[i], lstm_layer_category_outputs[i]], dim=-1)
            final_category_output = fc(category_output)
            final_category_outputs.append(final_category_output)

            sentiment_output = torch.cat([embedding_layer_sentiment_outputs[i], lstm_layer_sentiment_outputs[i]], dim=-1)
            final_sentiment_output = self.sentiment_fc(sentiment_output)
            final_sentiment_outputs.append(final_sentiment_output)

        if label is not None:
            category_labels = []
            polarity_labels = []
            polarity_masks = []
            for i in range(self.category_num):
                category_labels.append(label[:, i]) #term [:,] for batch
                polarity_labels.append(label[:, i + self.category_num])
                polarity_masks.append(polarity_mask[:, i]) #mask have same shape with label (just for category)

            loss = 0
            for i in range(self.category_num):
                category_temp_loss = self.category_loss(final_category_outputs[i].squeeze(dim=-1), category_labels[i])
                sentiment_temp_loss = self.sentiment_loss(final_sentiment_outputs[i], polarity_labels[i].long())
                loss += category_temp_loss
                loss += sentiment_temp_loss

            # # sentiment accuracy
            sentiment_logit = torch.cat(final_sentiment_outputs)
            sentiment_label = torch.cat(polarity_labels)
            sentiment_mask = torch.cat(polarity_masks)
            # self._accuracy(sentiment_logit, sentiment_label, sentiment_mask)

            # category f1
            final_category_outputs_prob = [torch.sigmoid(e) for e in final_category_outputs]
            category_prob = torch.cat(final_category_outputs_prob).squeeze()
            category_label = torch.cat(category_labels)
            self._f1(category_prob, category_label)
            category_metric=self._f1.get_metric()

            # output['loss'] = loss
        output = {
            'pred_category': [torch.sigmoid(e) for e in final_category_outputs],
            'pred_sentiment': [torch.softmax(e, dim=-1) for e in final_sentiment_outputs]
        }
        
        return output, loss

vocab=tokenizer.get_vocab()
vocab_size=len(vocab)
categories=aspect2idx.keys()
polarities=sentiment2idx.keys()
embedding_dim=300

word_embedder = torch.nn.Embedding(vocab_size, embedding_dim)
lr=3e-4
model = Cae(word_embedder, categories, polarities)
optimizer=torch.optim.Adam(model.parameters(), lr=lr)

epochs=1
for epoch in range(epochs):
    model.train()
    total_loss=0
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{epochs}"):
        batch_size = train_dataloader.batch_size
        polarity_mask = torch.ones(batch_size, num_aspect).float()
        input=batch['input_ids']
        attention_mask=batch['attention_mask']
        labels=batch['labels']
        output, loss = model(input, labels, polarity_mask,attention_mask)
        loss.backward()
        optimizer.step()
        total_loss+=loss
        break
    avg_loss=total_loss/len(train_dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")
torch.save(model.state_dict(), "CAE.pth")



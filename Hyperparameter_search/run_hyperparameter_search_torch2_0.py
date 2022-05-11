import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import warnings
warnings.filterwarnings("ignore")

def no_emoji(X):
    for i in range(len(X)):
        s = ''
        count = 0
        for j in range(len(X[i])):
            if X[i][j] == "[":
                count += 1
            elif count == -1:
                s += X[i][j]
            if X[i][j] == "]" and count > 0:
                count -= 1
        X[i] = s
    return X

def split(df, need_emoji = True, random_state = 0):
    X = list(df['review'])
    y = list(df['label'])
    # 60% train, 20% development, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = random_state)
    if not need_emoji:
        X_train = no_emoji(X_train)
        X_val = no_emoji(X_val)
        X_test = no_emoji(X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test

def compute_metrics(preds, labels):
    neg_labels = (labels == 0)
    neg_preds = (preds == 0)
    tp = sum(labels * preds).cpu().item()
    fp = sum(neg_labels * preds).cpu().item()
    fn = sum(labels * neg_preds).cpu().item()
    tn = len(preds) - tp - fp - fn
    return Counter({'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp})

def eval_model(model, data_loader, name) :
    model.eval()
    pbar = tqdm(total=len(data_loader))
    res = Counter({'tn':0, 'fp':0, 'fn':0, 'tp':0})
    for batch in data_loader:
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        res += compute_metrics(predictions, batch['labels'])
        pbar.update(1)
    pbar.close()
    print(f'{name} : {res}')

class WeiboSentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

df = pd.read_csv('../Data/processed_Weibo_data.csv')
df = df.dropna().drop("Unnamed: 0", axis = 1)
df.head()

df_emoji = df
# data with emoji
X_train_1, X_val_1, X_test_1, y_train_1, y_val_1, y_test_1 = split(df_emoji)
# same data with emoji removed
X_train_2, X_val_2, X_test_2, y_train_2, y_val_2, y_test_2 = split(df_emoji, need_emoji = False)


tokenizer = AutoTokenizer.from_pretrained('uer/chinese_roberta_L-12_H-768')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# data encodings
X_train_encodings = tokenizer(X_train_1, truncation=True, padding=True, return_tensors="pt").to(device)
X_val_encodings = tokenizer(X_val_1, truncation=True, padding=True, return_tensors="pt").to(device)
X_test_encodings = tokenizer(X_test_1, truncation=True, padding=True, return_tensors="pt").to(device)

y_train_1 = torch.tensor(y_train_1).to(device)
y_val_1 = torch.tensor(y_val_1).to(device).to(device)
y_test_1 = torch.tensor(y_test_1).to(device)

# dataset
train_dataset = WeiboSentDataset(X_train_encodings, y_train_1)
val_dataset = WeiboSentDataset(X_val_encodings, y_val_1)
test_dataset = WeiboSentDataset(X_test_encodings, y_test_1)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

lrs = [np.logspace(-5, -4.3, 5)[0]]
for lr in lrs:
    model = AutoModelForSequenceClassification.from_pretrained("uer/chinese_roberta_L-12_H-768")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, eps=1e-6, betas=(0.9,0.98), weight_decay=0.1)

    num_epochs = 3
    warmup_ratio = 0.1
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, warmup_ratio * num_training_steps, num_training_steps)

    pbar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        model.train()
        print(f'Epoch: {epoch}')
        for batch in train_loader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            pbar.update(1)
        
        eval_model(model, train_loader, 'Train')
        eval_model(model, val_loader, 'Validation')
        eval_model(model, test_loader, 'Test')
        torch.save(model.state_dict(), f'./models/epoch={epoch}_lr={lr}_.pth')
        
    pbar.close()
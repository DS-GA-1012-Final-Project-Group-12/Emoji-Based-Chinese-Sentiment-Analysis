import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers import Trainer, TrainingArguments

from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch

import logging
logging.basicConfig(level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

def no_emoji(X):
    for i in range(len(X)):
        s = ''
        count = 0
        for j in range(len(X[i])):
            if X[i][j] == "[":
                count += 1
            elif count == 0:
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

df = pd.read_csv('../Data/processed_Dianping_data.csv')
df = df.dropna().drop("Unnamed: 0", axis = 1)
df.head()

df_emoji = df
# data with emoji
X_train_1, X_val_1, X_test_1, y_train_1, y_val_1, y_test_1 = split(df_emoji)
# same data with emoji removed
X_train_2, X_val_2, X_test_2, y_train_2, y_val_2, y_test_2 = split(df_emoji, need_emoji = False)

class WeiboSentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


tokenizer = AutoTokenizer.from_pretrained('uer/chinese_roberta_L-12_H-768')

# data encodings
X_train_encodings = tokenizer(X_train_1, truncation=True, padding=True)
X_val_encodings = tokenizer(X_val_1, truncation=True, padding=True)
X_test_encodings = tokenizer(X_test_1, truncation=True, padding=True)

# dataset
train_dataset = WeiboSentDataset(X_train_encodings, y_train_1)
val_dataset = WeiboSentDataset(X_val_encodings, y_val_1)
test_dataset = WeiboSentDataset(X_test_encodings, y_test_1)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

#loop
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
optimizer = Adam(model.parameters(), lr=2e-5, adam_epsilon=1e-6, adam_beta1=0.9, adam_beta2=0.98)

num_epochs = 0.1
warmup_ratio = 0.1
num_training_steps = num_epochs * len(train_dataset)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, warmup_ratio * num_training_steps, num_training_steps, weight_decay=0.1)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device);

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        break


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
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

df_emoji = df[df['has_emoji'] == 1]
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

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    lr_scheduler_type='get_linear_schedule_with_warmup', # learning rate scheduler
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_ratio=0.1,                # number of warmup steps for learning rate scheduler
    #learning_rate=2e-5,              # learning rate or step size
    adam_epsilon=1e-6,
    adam_beta1=0.9,
    adam_beta2=0.98,
    weight_decay=0.01,
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,
)


def hp_space(trial) :
    return {
        "learning_rate": tune.loguniform(1e-5, 5e-5)
    }

def model_init() :
    return AutoModelForSequenceClassification.from_pretrained('uer/chinese_roberta_L-12_H-768')

def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    accuracy = accuracy_score(labels, preds)
    return {"accuracy":accuracy, 'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp}

trainer = Trainer(
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    model_init=model_init,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# Bayesian Optimisation
print(
    trainer.hyperparameter_search(
        hp_space=hp_space,
        compute_objective=lambda x:x["eval_accuracy"],
        n_trials=10,
        direction="maximize",
        backend="ray",
        search_alg=BayesOptSearch(), 
        mode="max", 
        local_dir="/scratch/ryl7673/NLU/NLU_PROJECT/Hyperparameter_search/models"
    )
)
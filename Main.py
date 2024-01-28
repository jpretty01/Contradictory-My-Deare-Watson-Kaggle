# Jeremy Pretty
#Contradictory My Dear Watson Kaggle
# Jan 28, 2024
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Constants
MODEL_NAME = 'bert-base-multilingual-cased'
BATCH_SIZE = 16
EPOCHS = 3
MAX_LEN = 128  # Max length of the tokenized input sequence

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Tokenizer for mBERT
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Dataset Class
class NLIDataset(Dataset):
    def __init__(self, premises, hypotheses, labels=None):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        premise = self.premises[idx]
        hypothesis = self.hypotheses[idx]
        encoding = tokenizer.encode_plus(
            premise,
            hypothesis,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        inputs = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.labels is not None:
            inputs['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return inputs

# Prepare the dataset and dataloader
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df[['premise', 'hypothesis']].values.tolist(), 
    train_df['label'].values, 
    test_size=0.1
)
train_dataset = NLIDataset([text[0] for text in train_texts], [text[1] for text in train_texts], train_labels)
val_dataset = NLIDataset([text[0] for text in val_texts], [text[1] for text in val_texts], val_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model = model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Validation Step
    model.eval()
    val_accuracy = []
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).flatten()
        accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        val_accuracy.append(accuracy)
    print(f"Epoch {epoch + 1}: Validation Accuracy: {sum(val_accuracy)/len(val_accuracy)}")

# Prediction on Test Data
test_dataset = NLIDataset([text[0] for text in test_df[['premise', 'hypothesis']].values.tolist()], [text[1] for text in test_df[['premise', 'hypothesis']].values.tolist()])
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
model.eval()
test_predictions = []
for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).flatten()
    test_predictions.extend(predictions.cpu().numpy())

# Prepare the submission file
submission_df = pd.DataFrame({'id': test_df['id'], 'label': test_predictions})
submission_df.to_csv('submission.csv', index=False)

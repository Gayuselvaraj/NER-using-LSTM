# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Identification of named entities is a major task in Natual Language Processing. To perform Named Entity Recognition, BiDirectional LSTM (Long Short Term Memory) can be used. The dataset used has various sentences along with its words and the corresponding tags. The task is to build a model that can classify these words into the right tags.

## DESIGN STEPS
## STEP 1:
Import required libraries in python to build your model and train it. Load your dataset and store required fields into different variables.

## STEP 2:
Allocate indexes for each unique word. Groud the words of each sentences and add the unique indexes for the words of each sentence. Split the dataset for training and testing.

## STEP 3:
Build your BiDirectional LSTM Model. It consists of embedding layer to convert the indexes of each word into vectors. Use dropout method to prevent overfitting. Specify the lstm and linear layer.

## PROGRAM
### Name:GAYATHRI S 
### Register Number:212224230073
```python
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128, num_layers=1, dropout=0.3):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=word2idx["ENDPAD"])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)  # *2 because bidirectional
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        out = self.fc(x)
        return out
        
```

```
vocab_size = len(word2idx)
tagset_size = len(tag2idx)

model = BiLSTMTagger(vocab_size, tagset_size).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=tag2idx["O"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

```

```
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)  # shape: [batch, seq_len, tagset_size]
            outputs = outputs.view(-1, outputs.shape[-1])
            labels = labels.view(-1)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids)
                outputs = outputs.view(-1, outputs.shape[-1])
                labels = labels.view(-1)
                val_loss = loss_fn(outputs, labels)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
<img width="873" height="651" alt="image" src="https://github.com/user-attachments/assets/02e375ae-1882-4a97-b6ab-814c5ba1e8ce" />


### Sample Text Prediction
<img width="671" height="527" alt="image" src="https://github.com/user-attachments/assets/c22d1a9a-8d93-4dad-b69c-3a6e6feac2d0" />




## RESULT
The BiLSTM NER model achieved good accuracy in identifying entities like persons, locations, and organizations. It showed strong performance on frequent tags, with scope for improvement on rarer ones.

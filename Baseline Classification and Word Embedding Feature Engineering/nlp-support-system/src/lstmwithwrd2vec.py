import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_pipline import load_and_preprocess_data, tokenize_and_preprocess, vectorize_text ,DATA_PATH
from gensim.models import Word2Vec
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X, y = load_and_preprocess_data(DATA_PATH)
tokenized_data = [tokenize_and_preprocess(text, is_tfidf=False) for text in X]

word2vec_model = Word2Vec(
    tokenized_data, vector_size=225, window=5, min_count=1, workers=4
)

x_train, x_test, y_train, y_test = vectorize_text(
    tokenized_data, y, is_TFIDF=False, word2vec_model=word2vec_model
)

# ── Label Encoding ───────────────────────────────────────────────────────────────
label_to_idx = {label: idx for idx, label in enumerate(sorted(set(y_train)))}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

y_train_encoded = [label_to_idx[label] for label in y_train]
y_test_encoded  = [label_to_idx[label] for label in y_test]

print(x_train.shape)

x_train_tensor = torch.tensor(x_train, dtype=torch.float).unsqueeze(1)  # (n_emails, seq_len, 100)
x_test_tensor  = torch.tensor(x_test,  dtype=torch.float).unsqueeze(1)

input_size  = word2vec_model.wv.vector_size  # 100 (Word2Vec vector size)
sequence_length = x_train_tensor.size(1)  # Assuming (n_emails, seq_len, 100)
hidden_size = 512
layers = 10
output_size = len(label_to_idx)

class RNN(nn.Module):
    def __init__(self,input_size, hidden_size,num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,       # input shape: (batch, seq_len, input_size)
            dropout=0.3             
        )
        self.fc       = nn.Linear(hidden_size, num_classes)
        #self.softmax  = nn.LogSoftmax(dim=1)  # ✅ Required for NLLLoss
        
    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])         # Take last time step 
        return out

print(f"x_train shape: {np.array(x_train).shape}") 


model     = RNN(input_size, hidden_size, layers, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)

train_losses = []
train_accs   = []
BATCH_SIZE = 1024  # Process one email at a time
for epoch in range(20):
    model.train()
    total_loss    = 0
    correct       = 0

    for i in range(0, x_train_tensor.size(0), BATCH_SIZE):       # ✅ Loop over emails
        
        x_batch = x_train_tensor[i:i+BATCH_SIZE].to(device)      # (batch, 1, 100)
        y_batch = torch.tensor(
            y_train_encoded[i:i+BATCH_SIZE], dtype=torch.long
        ).to(device)
        
        if x_batch.dim() == 2:
            x_batch = x_batch.unsqueeze(1)   # (batch, 100) → (batch, 1, 100)
        x_batch = x_batch.to(device)
        
        optimizer.zero_grad()

        # ✅ Single forward pass per email (averaged vector)
        output = model(x_batch)      # output: (batch_size, output_size)
        loss  = criterion(output, y_batch)  # NLLLoss expects (batch_size, output_size) and (batch_size)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (output.argmax(dim=1) == y_batch).sum().item()  # Compare predicted vs true labels

    avg_loss = total_loss / (x_train_tensor.size(0) / BATCH_SIZE)
    accuracy = correct   / x_train_tensor.size(0) * 100
    train_losses.append(avg_loss)
    train_accs.append(accuracy) 

    print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

def evaluate(model, x_tensor, y_encoded, batch_size=512):
    model.eval()
    correct = 0

    with torch.no_grad():
        for i in range(0, x_tensor.size(0), batch_size):
            x_batch = x_tensor[i:i+batch_size].to(device)
            y_batch = torch.tensor(
                y_encoded[i:i+batch_size], dtype=torch.long
            ).to(device)
            output   = model(x_batch)
            correct += (output.argmax(dim=1) == y_batch).sum().item()

    return correct / x_tensor.size(0) * 100


epochs = range(1, len(train_losses) + 1)

train_acc = evaluate(model, x_train_tensor, y_train_encoded)
test_acc = evaluate(model, x_test_tensor,  y_test_encoded)
print(f"\nFinal Train Accuracy : {train_acc:.2f}%")
print(f"Final Test  Accuracy : {test_acc:.2f}%")


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, 21), train_losses, marker='o', color='blue')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss")
ax1.grid(True)

ax2.plot(epochs,train_accs, marker='o', color='green')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training Accuracy")
ax2.grid(True)

plt.tight_layout()
plt.show()
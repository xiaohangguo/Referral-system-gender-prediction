import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
import numpy as np
num_train_chunks = 50
num_test_chunks = 10
num_epochs= 10
class CustomDataset(Dataset):
    def __init__(self, file_paths):
        self.data = []
        for file_path in file_paths:
            inputs, masks, labels = torch.load(file_path)
            self.data.extend(zip(inputs, masks, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids, attention_mask, labels = self.data[idx]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels - 1  # 将标签减去1
        }

# 数据文件路径
train_file_paths = [f'train_data/preprocessed_data_chunk_{i}.pt' for i in range(num_train_chunks)]
test_file_paths = [f'test_data/preprocessed_data_chunk_{i}.pt' for i in range(num_test_chunks)]

# 数据集和数据加载器
train_dataset = CustomDataset(train_file_paths)
test_dataset = CustomDataset(test_file_paths)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

# 初始化BERT模型
model = BertForSequenceClassification.from_pretrained('/public/home/lvshuhang/model_space/workspace/bert-base-uncased', num_labels=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        # print(loss)
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), 'bert_model.pt')

# 评估模型
model.eval()
predictions, true_labels = [], []
for batch in test_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = (batch['labels'] - 1).cpu().numpy()  # 将标签调整回原始范围
        predictions.extend(preds)
        true_labels.extend(labels)

accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {accuracy:.4f}")

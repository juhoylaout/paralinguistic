import torch.nn as nn
import torch.optim as optim
from transformers import AutoFeatureExtractor, Wav2Vec2ForXVector
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import torch
import os
import librosa
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
sampling_rate = 16_000
# feature_model= "facebook/wav2vec2-base"
feature_model= "facebook/wav2vec2-large-xlsr-53"
feature_extractor = AutoFeatureExtractor.from_pretrained(feature_model)

label_to_id = {'mask': 0, 'clear': 1}

def load_audio(batch):
    audio_files = batch['file_name']
    features = []
    labels = []
    
    for audio_file, label in zip(audio_files, batch["label"]):
        audio_path = os.path.join('./wav/', audio_file)
        audio, sr = librosa.load(audio_path, sr=sampling_rate)
        feature = feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            max_length=16000,
            feature_size=1,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
            truncation=True
        ).input_values.squeeze()
        features.append(feature.numpy())
        labels.append(label_to_id[label])
    
    batch['feature'] = features
    batch['labels'] = labels
    return batch

data_files = 'train_labels.csv'
raw_dataset = load_dataset('csv', data_files=data_files)
print("mapping training data...")
dataset = raw_dataset.map(load_audio, batched=True, batch_size=8, num_proc=1)


# Test data
test_data_files = 'devel_labels.csv'
raw_test_dataset = load_dataset('csv', data_files=test_data_files)
print("mapping test data...")
test_dataset = raw_test_dataset.map(load_audio, batched=True, batch_size=8, num_proc=1)

# Model Definition
class Wav2VecClassifier(nn.Module):
    def __init__(self, base_model, num_labels=11):
        super().__init__()
        self.base_model = base_model
        # self.hidden = nn.Linear(512, 512)
        self.classifier = nn.Linear(512, num_labels)
        self.base_model.freeze_feature_encoder()
        # self.base_model.freeze_base_model()
    def forward(self, input_values):

        outputs = self.base_model(input_values)
        embeddings = outputs.embeddings
        # embeddings = F.relu(self.hidden(embeddings))
        logits = self.classifier(embeddings)

        return logits
    
# Model Configuration
base_model = Wav2Vec2ForXVector.from_pretrained(
    feature_model,
    attention_dropout=0.05,
    activation_dropout=0.05,
    hidden_dropout=0.0,
    feat_proj_dropout=0.05,
    feat_quantizer_dropout=0.0,
    feat_extract_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.05,
    do_stable_layer_norm=True,
    apply_spec_augment=True
)

lr = 1e-4
batch_size = 16
num_warmup_steps = 800

num_epochs = 16

print("lr: ", lr)
print("batch_size: ", batch_size)
print("num_warmup_steps: ", num_warmup_steps)

classifier_model = Wav2VecClassifier(base_model, num_labels=len(label_to_id)).to(device)

# Training Setup
# optimizer = optim.AdamW(classifier_model.parameters(), lr=1e-4)
optimizer = optim.Adam(classifier_model.parameters(), 
                    lr=lr, 
                    betas=(0.9, 0.999), 
                    eps=1e-08)
criterion = nn.CrossEntropyLoss()

def collate_fn(batch):
    features = [torch.tensor(item['feature']) for item in batch]
    input_values = torch.stack(features)
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    return {'input_values': input_values, 'labels': labels}

# Train DataLoader
print("train loader...")
train_loader = DataLoader(dataset['train'], shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
# Test DataLoader
print("test loader...")
test_loader = DataLoader(test_dataset['train'], batch_size=batch_size, collate_fn=collate_fn)

# Warmup Schedule
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)


def train(epoch, model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    global_step = 0
    
    for i, batch in enumerate(loader):
        inputs = batch['input_values'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = classifier_model(inputs)

        #print("inputs: ", inputs.shape)
        #print("labels: ", labels)
        #print("logits: ", logits)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        global_step += 1
        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)

        correct_predictions += (predicted == labels).sum().item()
        
        if global_step % 10 == 0:
            print(f"[Epoch {epoch + 1}, Step {global_step}] loss: {running_loss / 10:.3f}")
            running_loss = 0.0

    acc = 100 * correct_predictions / len(loader.dataset)
    print(f"Accuracy on training set: {acc:.2f}%")


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = batch['input_values'].to(device)
            labels = batch['labels'].to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == labels).sum().item()

    loss = running_loss / len(loader.dataset)
    acc = 100 * correct_predictions / len(loader.dataset)
    print(f"Test loss: {loss:.3f}, Test accuracy: {acc:.2f}%")

    return loss, acc

# Training Loop
for epoch in range(num_epochs):
    train(epoch, classifier_model, train_loader, optimizer, criterion, device)
    loss, acc = evaluate(classifier_model, test_loader, criterion, device)

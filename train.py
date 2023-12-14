import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from spacefilling_vq import SpaceFillingVQ
from utils import codebook_initialization
import nemo.collections.asr as nemo_asr
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        "nvidia/speakerverification_en_titanet_large")


def load_audio(batch):
    audio_files = batch['file_name']
    features = []
    labels = []

    for audio_file, label in zip(audio_files, batch["L1"]):
        audio_path = os.path.join('./wav/', audio_file)
        emb = speaker_model.get_embedding(audio_path)

        features.append(emb)
        labels.append(label_to_id[label])

    batch['feature'] = features
    batch['labels'] = labels
    return batch


class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.3):
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.num_labels = num_labels
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, hidden_states, labels):
        labels = labels.flatten()
        weight = nn.functional.normalize(self.weight, dim=0)
        hidden_states = nn.functional.normalize(hidden_states, dim=1)
        cos_theta = torch.mm(hidden_states, weight)
        psi = cos_theta - self.margin

        onehot = nn.functional.one_hot(labels, self.num_labels)
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)
        loss = self.loss(logits, labels)

        return loss, cos_theta


# Model Definition
class Wav2VecClassifier(nn.Module):
    def __init__(self, base_model, num_labels=11):
        super().__init__()
        self.base_model = base_model
        # self.hidden = nn.Linear(512, 512)
        #self.classifier = nn.Linear(512, num_labels)
        self.classifier = nn.Sequential(nn.Linear(192, 512))
        self.objective = AMSoftmaxLoss(512, num_labels, scale=30.0, margin=0.2)
        self.base_model.freeze_feature_encoder()
        #self.base_model.freeze_base_model()
        
    def forward(self, input_values):

        #outputs = self.base_model(input_values)

        #embeddings = outputs.logits
        # embeddings = F.relu(self.hidden(embeddings))
        logits = self.classifier(input_values)

        return logits


class FullWav2VecClassifier(nn.Module):
    def __init__(self, quantizer):
        super().__init__()
        self.quantizer = quantizer
        self.classifier = nn.Sequential(nn.Linear(192, 1024), nn.ReLU(), nn.Linear(1024, 1024)).to(device)
        self.objective = AMSoftmaxLoss(1024, 11, scale=30.0, margin=0.2)

    def forward(self, input_values):
        input_values = torch.squeeze(input_values)
        #with torch.no_grad():
        #    embeddings = self.quantizer.evaluation(input_values)
        logits = self.classifier(input_values)

        return logits


def collate_fn(batch):
    features = [torch.tensor(item['feature']) for item in batch]
    input_values = torch.stack(features)
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    return {'input_values': input_values, 'labels': labels}


def train_classifier(epoch, model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    global_step = 0

    for i, batch in enumerate(loader):
        inputs = batch['input_values'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        hidden_pred = model(inputs)
        hidden_pred = torch.squeeze(hidden_pred)

        loss, logits = criterion(hidden_pred, labels)

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

            hidden_pred = model(inputs)
            hidden_pred = torch.squeeze(hidden_pred)
            loss, logits = criterion(hidden_pred, labels)

            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == labels).sum().item()

    loss = running_loss / len(loader.dataset)
    acc = 100 * correct_predictions / len(loader.dataset)
    print(f"Eval loss: {loss:.3f}, Eval accuracy: {acc:.2f}%")

    return loss, acc


def test(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0

    predictions_list = []
    labels_list = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = batch['input_values'].to(device)
            labels = batch['labels'].to(device)

            hidden_pred = model(inputs)
            hidden_pred = torch.squeeze(hidden_pred)
            loss, logits = criterion(hidden_pred, labels)

            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == labels).sum().item()

            predictions_list.append(predicted.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())

    loss = running_loss / len(loader.dataset)
    acc = 100 * correct_predictions / len(loader.dataset)
    print(f"Test loss: {loss:.3f}, Test accuracy: {acc:.2f}%")

    return loss, acc, predictions_list, labels_list

# Training Loop


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sampling_rate = 16000

    label_to_id = {
        'ARA': 0, 'CHI': 1, 'FRE': 2, 'GER': 3, 'HIN': 4,
        'ITA': 5, 'JPN': 6, 'KOR': 7, 'SPA': 8, 'TEL': 9, 'TUR': 10
    }

    # Test Data
    test_data_files = './lab/test.tsv'
    raw_test_dataset = load_dataset('csv', data_files=test_data_files, delimiter='\t')
    print("mapping test data...")
    test_dataset = raw_test_dataset.map(load_audio, batched=True, batch_size=8, num_proc=1)

    # Train Data
    data_files = './lab/train.tsv'
    raw_dataset = load_dataset('csv', data_files=data_files, delimiter='\t')
    print("mapping training data...")
    dataset = raw_dataset.map(load_audio, batched=True, batch_size=8, num_proc=1)

    # Eval data
    eval_data_files = './lab/eval.tsv'
    raw_test_dataset = load_dataset('csv', data_files=eval_data_files, delimiter='\t')
    print("mapping eval data...")
    eval_dataset = raw_test_dataset.map(load_audio, batched=True, batch_size=8, num_proc=1)


    # Model Configuration
    lr = 0.001
    batch_size = 64
    num_warmup_steps = 400

    num_epochs = 40

    # Vector Quantization Init
    desired_vq_bitrate = 9
    embedding_dim = 192
    initial_codebook = codebook_initialization(torch.randn(int(1e4), embedding_dim)).to(device)
    learning_rate = 0.001

    print("lr: ", lr)
    print("batch_size: ", batch_size)
    print("num_warmup_steps: ", num_warmup_steps)

    # Train DataLoader
    print("train loader...")
    train_loader = DataLoader(dataset['train'], shuffle=True, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
    # Eval Dataloader

    print("eval loader...")
    eval_loader = DataLoader(eval_dataset['train'], batch_size=batch_size, collate_fn=collate_fn, num_workers=0)

    # Test DataLoader
    print("test loader...")
    test_loader = DataLoader(test_dataset['train'], batch_size=batch_size, collate_fn=collate_fn, num_workers=0)

    # Warmup Schedule
    total_steps = len(train_loader) * num_epochs

    # VQ Init
    vector_quantizer = SpaceFillingVQ(desired_vq_bitrate, embedding_dim, device=device,
                                      initial_codebook=initial_codebook)
    vector_quantizer.to(device)

    checkpoint = torch.load(f"vector_quantizer_{desired_vq_bitrate}bits_bs{64}_lr{learning_rate}.pt")
    vector_quantizer.load_state_dict(checkpoint["vector_quantizer"])

    full_model = FullWav2VecClassifier(vector_quantizer).to(device)

    # Training Setup
    optimizer = optim.AdamW(full_model.parameters(), lr=lr)
    criterion = full_model.objective

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)

    acc_best = 0
    for epoch in range(num_epochs):
        train_classifier(epoch, full_model, train_loader, optimizer, criterion, device)
        loss, acc = evaluate(full_model, eval_loader, criterion, device)

        if acc >= acc_best:
            torch.save(full_model.state_dict(), 'best_model_native_language_VQ.pth')
            acc_best = acc

    loss, acc, predictions, actual = test(full_model, test_loader, criterion, device)

    confusion_matrix = metrics.confusion_matrix(actual, predictions)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

    cm_display.plot()
    plt.show()

    print(f"Final Test Accuracy: {acc:.2f}%")

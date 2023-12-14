"""
An example code to show how to train the Space-FillingVQ module on a Normal distribution. Notice that the bitrate
for Space-FillingVQ has to be increased step by step during training, starting from 2 bits (4 corner points) to
desired bitrate (2**desired_bits corner points).
"""

import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import torch
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F

from spacefilling_vq import SpaceFillingVQ
from utils import codebook_initialization, codebook_extension
import nemo.collections.asr as nemo_asr


# Hyper-parameters
desired_vq_bitrate = 9
codebook_extension_eps = 0.01
embedding_dim = 192

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
sampling_rate = 16_000


label_to_id = {
    'ARA': 0, 'CHI': 1, 'FRE': 2, 'GER': 3, 'HIN': 4,
    'ITA': 5, 'JPN': 6, 'KOR': 7, 'SPA': 8, 'TEL': 9, 'TUR': 10
}

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")

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

def collate_fn(batch):
    features = [torch.tensor(item['feature']) for item in batch]
    input_values = torch.stack(features)
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    return {'input_values': input_values, 'labels': labels}

batch_size = 64

# Train DataLoader
print("train loader...")
train_loader = DataLoader(dataset['train'], shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
# Eval Dataloader

print("eval loader...")
eval_loader = DataLoader(eval_dataset['train'], batch_size=batch_size, collate_fn=collate_fn)

# Test DataLoader
print("test loader...")
test_loader = DataLoader(test_dataset['train'], batch_size=batch_size, collate_fn=collate_fn)

num_epochs = 10
learning_rate = 0.001
num_batches = int(len(train_loader) / batch_size)
milestones = [int(num_epochs*0.6), int(num_epochs*0.8)]

# Arrays to save the logs of training
total_vq_loss = np.zeros((desired_vq_bitrate - 1, num_epochs)) # tracks VQ loss
total_perplexity = np.zeros((desired_vq_bitrate - 1, num_epochs)) # tracks perplexity
used_codebook_indices_list = [] # tracks indices of used codebook entries

initial_codebook = codebook_initialization(torch.randn(int(1e4), embedding_dim)).to(device)

vector_quantizer = SpaceFillingVQ(desired_vq_bitrate, embedding_dim, device=device, initial_codebook=initial_codebook)
vector_quantizer.to(device)

for bitrate in range(2, desired_vq_bitrate+1):

    optimizer = optim.Adam(vector_quantizer.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    entries = int(2 ** bitrate) # Current bitrate for Space-FillingVQ (current number of corner points)
    used_codebook_indices = np.zeros((entries,))

    if bitrate > 2: # Codebook extension phase (increasing sapce-FillingVQ bitrate/corner points)
        final_indices = codebook_extension(vector_quantizer.entries, codebook_extension_eps).to(device)
        codebook = vector_quantizer.decode(final_indices)
        vector_quantizer.codebook.data[0:int(2**bitrate)] = codebook

    for epoch in range(num_epochs):

        vq_loss_accumulator = perplexity_accumulator = 0

        print(f'<<<<<<<<<<########## VQ Bitrate = {bitrate} | Epoch = {epoch + 1} ##########>>>>>>>>>>')

        for i, batch in enumerate(train_loader):

            data = batch['input_values'].to(device)
            data = torch.squeeze(data)

            optimizer.zero_grad()

            quantized, perplexity, selected_indices = vector_quantizer(data, entries)

            vq_loss = F.mse_loss(data, quantized) # use this loss if you are exclusively training only the
                                                        # Space-FillingVQ module.

            vq_loss.backward()
            optimizer.step()

            used_codebook_indices[selected_indices] += 1
            used_codebook_indices[selected_indices+1] += 1

            vq_loss_accumulator += vq_loss.item()
            perplexity_accumulator += perplexity.item()

            vq_loss_average = vq_loss_accumulator / (i+1)
            perplexity_average = perplexity_accumulator / (i+1)

        total_vq_loss[bitrate-2, epoch] = vq_loss_average
        total_perplexity[bitrate-2, epoch] = perplexity_average

        scheduler.step()

        # printing the training logs for each epoch
        print("epoch:{}, vq loss:{:.6f}, perpexlity:{:.4f}".format(epoch+1, vq_loss_average, perplexity_average))

    used_codebook_indices_list.append(used_codebook_indices)

# saving the training logs and Space-FillingVQ trained model
np.save(f'total_vq_loss_{desired_vq_bitrate}bits_bs{batch_size}_lr{learning_rate}.npy', total_vq_loss)
np.save(f'total_perplexity_{desired_vq_bitrate}bits_bs{batch_size}_lr{learning_rate}.npy', total_perplexity)

with open(f"used_codebook_indices_list_{desired_vq_bitrate}bits_bs{batch_size}_lr{learning_rate}", "wb") as fp:
    pickle.dump(used_codebook_indices_list, fp)

checkpoint_state = {"vector_quantizer": vector_quantizer.state_dict()}
torch.save(checkpoint_state, f"vector_quantizer_{desired_vq_bitrate}bits_bs{batch_size}_lr{learning_rate}.pt")

print("\nTraining Finished >>> Logs and Checkpoints Saved!!!")

######################## Evaluation (Inference) of Space-FillingVQ #############################

quantized_data = torch.zeros_like(data)

eval_batch_size = 64
num_batches = int(data.shape[0]/eval_batch_size)
with torch.no_grad():
    for i in range(num_batches):
        data_batch = data[(i*eval_batch_size):((i+1)*eval_batch_size)]
        quantized_data[(i*eval_batch_size):((i+1)*eval_batch_size)] = vector_quantizer.evaluation(data_batch)

mse = F.mse_loss(data, quantized_data).item()
print("Mean Squared Error = {:.4f}".format(mse))


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
import torch.utils.data as data
import math
import numpy as np
import os

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

model_path = './models/'
embed_size = 256
hidden_size = 512
learning_rate = 0.001
num_epochs = 5
batch_size = 128

# image preprocessing
transform = transforms.Compose([ 
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), 
                         (0.229, 0.224, 0.225))])

# build data loader
data_loader, caption_lengths = get_loader(transform=transform,
                                          batch_size=batch_size)
vocab_size = len(data_loader.dataset.vocab)

# build the architectures
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()

criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

params = list(decoder.parameters()) + list(encoder.linear.parameters()) 
optimizer = torch.optim.Adam(params=params, lr=learning_rate)

total_step = math.ceil(len(caption_lengths) / data_loader.batch_sampler.batch_size)

if not os.path.exists(model_path):
    os.makedirs(model_path)

for epoch in range(num_epochs):
    
    i_step = 0
    
    while i_step < total_step:

        sel_length = np.random.choice(caption_lengths)
        all_indices = np.where([caption_lengths[i] == sel_length for i in np.arange(len(caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=batch_size))
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler

        for batch in data_loader:
            images, captions = batch[0], batch[1]
            break 

        images = to_var(images, volatile=True)
        captions = to_var(captions)

        decoder.zero_grad()
        encoder.zero_grad()
        features = encoder(images)
        outputs = decoder(features, captions)
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        loss.backward()
        optimizer.step()
            
        i_step += 1
        # print info
        print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
            %(epoch+1, num_epochs, i_step, total_step, loss.data[0], np.exp(loss.data[0]))) 
            
        if i_step >= total_step:
            break

    # save the models
    torch.save(decoder.state_dict(), os.path.join(model_path, 'decoder-%d.pkl' %(epoch+1)))
    torch.save(encoder.state_dict(), os.path.join(model_path, 'encoder-%d.pkl' %(epoch+1)))
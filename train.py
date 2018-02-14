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

batch_size = 128
vocab_threshold = 4
embed_size = 256
hidden_size = 512
learning_rate = 0.001
num_epochs = 5
save_every = 1

# image preprocessing
transform_train = transforms.Compose([ 
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), 
                         (0.229, 0.224, 0.225))])

# build data loader
data_loader = get_loader(transform=transform_train,
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold)

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

total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)

for epoch in range(num_epochs):
    
    for i_step in range(0, total_step):

        indices = data_loader.dataset.get_train_indices()
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
            
        # print info
        print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
            %(epoch+1, num_epochs, i_step, total_step, loss.data[0], np.exp(loss.data[0]))) 
            
    # save the weights
    if epoch % save_every == 0:
        torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-%d.pkl' %(epoch+1)))
        torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-%d.pkl' %(epoch+1)))
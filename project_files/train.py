import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
import torch.utils.data as data

model_path = './models/'
embed_size = 256
hidden_size = 512
learning_rate = 0.001
num_epochs = 5

# image preprocessing
transform = transforms.Compose([ 
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), 
                         (0.229, 0.224, 0.225))])

# build data loader
data_loader = get_loader(transform=transform)
vocab_size = len(data_loader.dataset.vocab)

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

# build the architectures
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()

criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

params = list(decoder.parameters()) + list(encoder.vgg16.classifier) 
optimizer = torch.optim.Adam(params=params, lr=learning_rate)

total_step = ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)

for epoch in range(num_epochs):
    for i in range(total_step):

        sel_length = np.random.choice(data_loader.dataset.caption_lengths)
        indices = data_loader.dataset.captions_dict[sel_length]
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler

        images, captions = next(iter(data_loader))

        images = to_var(images, volatile=True)
        captions = to_var(captions)

        decoder.zero_grad()
        encoder.zero_grad()
        features = encoder(images)
        outputs = decoder(features, captions)
        loss = criterion(outputs, captions)
        loss.backward()
        optimizer.step()

        # print info
        print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
            %(epoch, num_epochs, i, total_step, loss.data[0], np.exp(loss.data[0]))) 

    # save the models
    torch.save(decoder.state_dict(), os.path.join(args.model_path, 'decoder-%d.pkl' %(epoch+1)))
    torch.save(encoder.state_dict(), os.path.join(args.model_path, 'encoder-%d.pkl' %(epoch+1)))
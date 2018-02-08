import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

class EncoderCNN(nn.Module):
	def __init__(self, embed_size):
		super(EncoderCNN, self).__init__()
		self.vgg16 = models.vgg16(pretrained=True)
		in_features = list(self.vgg16.classifier.children())[0].in_features
		self.vgg16.classifier = nn.Linear(in_features, embed_size)

	def forward(self, images):
		features = self.vgg16(images)
		return features

class DecoderRNN(nn.Module):
	def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
		super(DecoderRNN, self).__init__()
		self.embed = nn.Embedding(vocab_size, embed_size)
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, vocab_size)

	def forward(self, features, captions, lengths):
		embeddings = self.embed(captions)
		inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
		packed = pack_padded_sequence(inputs, lengths, batch_first=True)
		hiddens, _ = self.lstm(packed)
		outputs = self.linear(hiddens[0])
		return outputs
import nltk
import os
import torch
import torch.utils.data as data
from torchvision import transforms
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO

sample_transform = transforms.Compose([ 
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

def get_loader(img_folder='images/train2014/', 
			   transform=sample_transform,
			   vocab_file='annotations/vocab.pkl',
			   pad_word="<pad>",
			   start_word="<start>",
			   end_word="<end>",
			   unk_word="<unk>",
			   vocab_threshold=4,
			   captions_file='annotations/captions_train2014.json',
			   batch_size=5,
			   shuffle=True,
			   num_workers=2):
	
	# COCO caption dataset
	coco = CoCoDataset(img_folder=img_folder,
					   transform=transform,
					   vocab_file=vocab_file,
					   pad_word=pad_word,
					   start_word=start_word,
					   end_word=end_word,
					   unk_word=unk_word,
					   vocab_threshold=vocab_threshold,
					   captions_file=captions_file)

	# data loader for COCO dataset
	data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                          	  batch_size=batch_size,
                                          	  shuffle=shuffle,
                                          	  num_workers=num_workers,
                                          	  collate_fn=collate_fn)
	return data_loader

class CoCoDataset(data.Dataset):
    
    def __init__(self, img_folder, transform, vocab_file, pad_word,
		start_word, end_word, unk_word, vocab_threshold, captions_file):
        self.img_folder = img_folder
        self.transform = transform
        self.coco = COCO(captions_file)
        self.ids = list(self.coco.anns.keys())
        self.vocab = Vocabulary(vocab_file, pad_word, start_word,
        	end_word, unk_word, vocab_threshold, captions_file)
        self.vocab.get_vocab()

    def __getitem__(self, index):
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab(self.vocab.start_word))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab(self.vocab.end_word))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths
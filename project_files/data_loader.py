import nltk
import os
import torch
import torch.utils.data as data
from torchvision import transforms
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm

def get_loader(transform,
               img_folder='images/train2014/', 
               vocab_file='annotations/vocab.pkl',
               pad_word="<pad>",
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_threshold=4,
               captions_file='annotations/captions_train2014.json',
               batch_size=1,
               num_workers=1):
    
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
                                              num_workers=num_workers,
                                              batch_sampler=data.sampler.BatchSampler(sampler=coco.sampler,
                                                                                      batch_size=batch_size,
                                                                                      drop_last=False)
                                              )
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
        
        # stuff
        all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
        self.caption_lengths = [len(token) for token in all_tokens]
        self.caption_index = 0
        self.get_next_sampler()

    def __getitem__(self, index):
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab(self.vocab.start_word))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab(self.vocab.end_word))
        caption = torch.Tensor(caption).long()
        return image, caption

    def __len__(self):
        return len(self.ids)

    def get_next_sampler(self):
        sampled_length = np.unique(self.caption_lengths)[self.caption_index]
        indices = list(np.where([self.caption_lengths[i] == sampled_length for i in np.arange(len(self.caption_lengths))])[0])
        self.caption_index = (self.caption_index + 1) % len(np.unique(self.caption_lengths))
        self.sampler = data.sampler.SubsetRandomSampler(indices=indices)

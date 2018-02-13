import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random

def get_loader(transform,
               batch_size,
               vocab_threshold,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               captions_file='../cocoapi/annotations/captions_train2014.json',
               vocab_from_file=True,
               img_folder='../cocoapi/resized_images/train2014/', 
               num_workers=2):
    
    # COCO caption dataset
    dataset = CoCoDataset(transform=transform,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          captions_file=captions_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    indices = dataset.get_train_indices()
    initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)

    # data loader for COCO dataset
    data_loader = data.DataLoader(dataset=dataset, 
                                  num_workers=num_workers,
                                  batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                          batch_size=dataset.batch_size,
                                                                          drop_last=False))

    return data_loader

class CoCoDataset(data.Dataset):
    
    def __init__(self, transform, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, captions_file, vocab_from_file, img_folder):
        self.transform = transform
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, captions_file, vocab_from_file)
        self.coco = COCO(captions_file)
        self.img_folder = img_folder
        self.ids = list(self.coco.anns.keys())
        self.get_caption_lengths()
        
    def __getitem__(self, index):
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        # Convert image to tensor and pre-process using transform
        image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        image = self.transform(image)

        # Convert caption to tensor of word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab(self.vocab.start_word))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab(self.vocab.end_word))
        caption = torch.Tensor(caption).long()
        
        return image, caption

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def get_random_item(self):
        ann_id = random.sample(self.ids, 1)[0]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        # Convert image to tensor and pre-process using transform
        image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)

        return image, image_tensor, caption

    def __len__(self):
        return len(self.ids)

    def get_caption_lengths(self):
        print('Obtaining caption lengths ...')
        all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
        self.caption_lengths = [len(token) for token in all_tokens]

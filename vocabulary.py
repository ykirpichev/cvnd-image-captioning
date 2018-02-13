import nltk
import pickle
import os.path
from pycocotools.coco import COCO
from collections import Counter

class Vocabulary(object):
    """Vocabulary class for an image-to-text model."""
    
    def __init__(self, 
        vocab_threshold=4, 
        vocab_file='./vocab.pkl', 
        pad_word="<pad>", 
        start_word="<start>",
        end_word="<end>", 
        unk_word="<unk>", 
        captions_file='../cocoapi/annotations/captions_train2014.json'):
        """Initialize the vocabulary.
        Args:
          vocab_file: File containing the vocabulary.
          pad_word: Special word denoting sentence padding.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
          vocab_threshold: Minimum word count threshold.
          captions_file: Path for train annotation file.
        """
        self.vocab_file = vocab_file
        self.pad_word = pad_word
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.vocab_threshold = vocab_threshold
        self.captions_file = captions_file
        self.get_vocab()

    def get_vocab(self):
        if not os.path.exists(self.vocab_file):
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)
        with open(self.vocab_file, 'rb') as f:
            vocab = pickle.load(f)
            self.word2idx = vocab.word2idx
            self.idx2word = vocab.idx2word

    def build_vocab(self):
        self.init_vocab()
        self.add_word(self.pad_word)
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def init_vocab(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        coco = COCO(self.captions_file)
        counter = Counter()
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if i % 1000 == 0:
                print("[%d/%d] Tokenized the captions." %(i, len(ids)))

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
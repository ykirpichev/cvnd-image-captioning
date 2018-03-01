# cvnd-image-captioning

## for workspaces

1. clone this repo: https://github.com/cocodataset/cocoapi
```
git clone https://github.com/cocodataset/cocoapi.git
```

2. setup (also described in the readme [here](https://github.com/cocodataset/cocoapi))
```
cd cocoapi/PythonAPI
make
cd ..
```

3. download some data from here: http://cocodataset.org/#download (described below)

     * under **Annotations**, download:
          - **2014 Train/Val annotations [241MB]** (extract `captions_train2014.json` and `captions_val2014.json`, and place at locations `cocoapi/annotations/captions_train2014.json` and `cocoapi/annotations/captions_val2014.json`, respectively)
          - **2014 Testing Image info [1MB]** (extract `image_info_test2014.json` and place at location `cocoapi/annotations/image_info_test2014.json`)
     * under **Images**, download:
          - **2014 Train images [83K/13GB]** (extract the `train2014` folder and place at location `cocoapi/images/train2014/`)
          - **2014 Val images [41K/6GB]** (extract the `val2014` folder and place at location `cocoapi/images/val2014/`)
          - **2014 Test images [41K/6GB]** (extract the `test2014` folder and place at location `cocoapi/images/test2014/`)

## rubric dump

must
- no modifications to data_loader.py or vocabulary.py
- when using data_loader.py to train the model, most arguments left at default value, as described in step 1 of 1_Preliminaries.ipynb
- implementation of CNN encoder passes test in step 3 of 1_Preliminaries.ipynb
- implementation of RNN decoder passes test in step 4 of 1_Preliminaries.ipynb
- transform chosen for training and testing phases make sense individually, and together
- trainable parameters make sense
- optimizer makes sense
- chosen hyperparameters have some evidence backing them re: why they are good choices (either from existing research papers or evidence where student demonstrated good performance)
- **2_Training.ipynb** is all cleaned up with a straightforward implementation that is easy to follow. like final code type thing
- step 3 of **3_Inference.ipynb** does not throw errors
- step 4 returns clean caption w/ no special tokens (except unknown token which may appear)
- all of the questions in **4_Retrospective.ipynb** are answered

above and beyond
- validation
- beam sampler
- train for long enough to compare to results in the literature. calculate BLEU score.
# cvnd-image-captioning

## Obtain the COCO Dataset

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

## Rubric (Draft)

#### Files Submitted

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Submission Files   | No modifications have been made to **data_loader.py** or **vocabulary.py**. |
| `CNNEncoder`  | The `CNNEncoder` class in **model.py** passes the test in **Step 3** of **1_Preliminaries.ipynb**.  The chosen CNN architecture makes sense as an encoder for the image captioning task. |
| `RNNDecoder`  | The `RNNDecoder` class in **model.py** passes the test in **Step 4** of **1_Preliminaries.ipynb**.  `RNNDecoder` class in **model.py** passes the test in **Step 3** of **3_Inference.ipynb**.  The chosen RNN architecture makes sense as a decoder for the image captioning task.  |
| Saved Models | The `models/` folder contains the final, trained weights for the CNN and RNN architectures. It contains two pickle files: one corresponds to the encoder, and the other corresponds to the decoder.  |


#### 2_Training.ipynb

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Using the Data Loader | When using the `get_loader` function in **data_loader.py** to train the model, most arguments are left at their default values, as outlined in **Step 1** of **1_Preliminaries.ipynb**.  In particular, the submission only (optionally) changes the values of the following arguments: `transform`, `mode`, `batch_size`, `vocab_threshold`, `vocab_from_file`. |
| **Step 1, Question 1** | The submission describes the chosen CNN-RNN architecture and details how the hyperparameters were selected. |
| **Step 1, Question 2** | The submission describes how the transform used to pre-process the training images was selected.  The transform is congruent with the choice of CNN architecture. |
| **Step 1, Question 3** | The submission describes how the trainable parameters were selected and has made a well-informed choice when deciding which parameters in the model should be trainable. |
| **Step 1, Question 4** | The submission describes how the optimizer was selected. |
| **Step 2** | The code cell in **Step 2** details all code used to train the model from scratch.  The output of the code cell shows exactly what is printed when running the code cell.  If the submission has amended the code used for training the model, it is well-organized and includes comments. |

#### 3_Inference.ipynb

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| `transform_test` | The transform used to pre-process the test images is congruent with the choice of CNN architecture.  It is also consistent with the transform specified in `transform_train` in **2_Training.ipynb**. | 
| **Step 3** | The implementation of the `sample` method in the `RNNDecoder` class correctly leverages the RNN to generate predicted token indices. |
| **Step 4** | The `clean_sentence` function passes the test in **Step 4**.  The sentence is reasonably clean, where any `<start>` and `<end>` tokens have been removed.  | 
| **Step 5** | The submission shows two image-caption pairs where the model performed well, and two image-caption pairs where the model did not perform well. |


## Suggestions to Make your Project Stand Out!

- Use the validation set to guide your search for appropriate hyperparameters.

-  Implement beam search to generate captions on new images.

- Tinker with your model - and train it for long enough - to obtain results that are comparable to (or surpass!) recent research articles.

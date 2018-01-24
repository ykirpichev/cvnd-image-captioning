# cvnd-image-captioning

just notes to myself that might also help other CDs get up and running with the COCO dataset. (_students will receive guidance in a very, very different format :)_)

## getting coco api up and running

1. clone this repo: https://github.com/cocodataset/cocoapi
```
git clone https://github.com/cocodataset/cocoapi.git
```

2. setup (as described in the readme [here](https://github.com/cocodataset/cocoapi))
```
cd cocoapi/PythonAPI
make
cd ..
```

3. get some data from here: http://cocodataset.org/#download ... 

     * under Annotations, download **2014 Train/Val annotations [241MB]**.  extract.  

     * this produces a folder `annotations`.  place it at location `cocoapi/annotations`.
  
4. place the `play_with_API.ipynb` notebook in this repo at location `cocoapi/play_with_API.ipynb`.

     * you should be good to go to explore the coco dataset with this file (i.e., all the cells should run without error).  if not, hit me up on slack @alexis.  

     * for more details, check out the documentation [here](http://cocodataset.org/#download). the ipynb is adapted from [this file](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb)


## generating the vocab

now, we can use the captions in the training set to generate a lookup table that maps words to indices (and vice versa)

1. create a `data` directory in `cocoapi`
```
mkdir data
```

2. place the jupyter notebook `build_vocab.ipynb`  in this repo at location `cocoapi/build_vocab.ipynb`, and run all cells

     * this will generate a file `data/vocab.pkl` that we'll use soon for processing captions. 

## preprocessing images

1. get some MORE data from here: http://cocodataset.org/#download 

     * under Images, download: **2014 Train images [83K/13GB]** and **2014 Val images [41K/6GB]**.  

     * extract both folders and place them at location `cocoapi/images/train2014` and `cocoapi/images/val2014`.

     * dataset is so big already :-O that we won't worry about the testing set for now.

2. place the jupyter notebook `play_with_api.ipynb`  in this repo at location `cocoapi/play_with_api.ipynb`, and run all cells

     * change the settings to visualize the effect of different options

the main idea behind this notebook // i have not completely decided the best way to resize ... so created a playground to experiment a bit, to see what makes the most sense. i'm currently considering a fe different options, that involve two distinct choices:
- **CHOICE 1**: how to resize images? options:
    - resize all images to 256x256 (ignore aspect ratio)
    - resize all images to be 256 in smaller edge
    - resize all images to 224x224 (ignore aspect ratio)
    - resize all images to be 224 in smaller edge
- **CHOICE 2**: how to crop? options:
    - do random 224x224 crop (helps with generalization but could crop out crucial part of the image if not careful)
    - do center 224x224 crop

## setting up the data loader

1. more instructions coming soon to a repo near you

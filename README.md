# cvnd-image-captioning

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

3. get some data from here: http://cocodataset.org/#download ... under Annotations, download **2014 Train/Val annotations [241MB]**.  extract.  this produces a folder `annotations`.  place it at location `cocoapi/annotations`.
  
4. place the jupyter notebook in this repo at location `cocoapi/play_with_API.ipynb`.  you should be good to go to explore the coco dataset with this file (i.e., all the cells should run without error).  if not, hit me up on slack @alexis.  for more details, check out the documentation here: http://cocodataset.org/#download. (the ipynb is adapted from this file: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb)

5. if you want to start training in pytorch, you'll need to also download the images ... under Images, download: (1) **2014 Train images [83K/13GB]**, (2) **2014 Val images [41K/6GB]**, and (3) **2014 Test images [41K/6GB]**.  extract all three folders and place them at location `cocoapi/images/train2014`, `cocoapi/images/val2014`, `cocoapi/images/test2014`

#### coming soon to a repo near you: preprocessing for pytorch ...

# cvnd-image-captioning

just notes to myself that might also help other CDs get up and running with the COCO dataset. (_students will receive guidance in a very, very different format :)_)

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

3. get some data from here: http://cocodataset.org/#download ... 

     * under **Annotations**, download:
          - **2014 Train/Val annotations [241MB]** (extract `captions_train2014.json` and `captions_val2014.json`, and place at locations `cocoapi/annotations/captions_train2014.json` and `cocoapi/annotations/captions_val2014.json`, respectively)
          - **2014 Testing Image info [1MB]** (extract `image_info_test2014.json` and place at location `cocoapi/annotations/image_info_test2014.json`)
     * under **Images**, download:
          - **2014 Train images [83K/13GB]**
          - **2014 Val images [41K/6GB]**
          - **2014 Test images [41K/6GB]**

## getting coco api up and running (if you don't wanna download everything)

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



# ssd-keras
A implementation of SSD net by using keras.
Now is the simple implementation of 10-class number identification.

Software requirement:
```
keras > 2.1.5
tensorflow-gpu < 2.0
```

Directory structure:
```
│  .gitignore
│  predict.py               # predict image
│  train.py                 # train the model
├─img                       # contain test image
├─layers                    # store custom layers
│      normalized.py
│      prior_box_layer.py
│      __init__.py
├─model_data
|      prior_boxes_ssd300.pkl   # store prior boxes, need to download
├─networks                  # store network structure
│      model.png
│      ssd_linked.py
│      ssd_straight.py
│      vgg16.py
│      __init__.py
├─parameter                 # store parameters and hyper parameters
│      parameters.py
│      __init__.py
├─tff
│      simhei.ttf
├─util
│      decode_util.py
│      drawing_util.py
│      img_process_util.py
|      data_util.py
│      __init__.py
└─weight                    # store weights, need to download
        ssd_weights.h5
```

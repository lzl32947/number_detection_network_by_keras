# ssd-keras
A implementation of SSD net by using keras.

Software requirement:
```
keras > 2.1.5
tensorflow-gpu < 2.0
```

Directory structure:
```
│  .gitignore
│  predict.py               # predict image
├─img                       # contain test image
│      people.jpg
│      street.jpg
│      street2.jpg
├─layers                    # store custom layers
│      normalized.py
│      prior_box_layer.py
│      __init__.py
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
│      __init__.py
└─weight                    # store weights, but need to download
        ssd_weights.h5
```

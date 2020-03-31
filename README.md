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
├─config                # Store configurations
├─data                  # Data for train and test
├─log               
│  ├─checkpoint         # Checkpoint dir
│  ├─tensorboard        # tensorboard log dir
│  └─weight             # Pre-trained weight dir
├─model                 # Module of used model
│  ├─image              # The image of models
│  └─layers             # Custom layers
├─other                 # Included files
│  ├─font               # Font file for ImageDraw
│  └─prior_box          # Pre-generated numpy array
└─util                  # Utils
```

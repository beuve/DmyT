# DmyT: Dummy Triplet Loss for Deepfake Detection

![](./docs/header.png)

This is a Pytorch implementation of "DmyT: Dummy Triplet Loss for Deepfake Detection"

### Prepare your dataset

Your need to organise your data as follow:

```
dataset
├── train
│   ├── fake
|   |   ╰ ...
│   ╰── real
|       ╰ ...
├── test
│   ├── fake
|   |   ╰ ...
│   ╰── real
|       ╰ ...
╰── valid
    ├── fake
    |   ╰ ...
    ╰── real
        ╰ ...
```

### Training

Training can be performed using the script `src/train.py`. You can use this bash script and complete the parameters

```bash
# Learning rate (ex: 2e-5)
LR = 
# Number of epochs
EPOCH = 
# Model name from TIMM library
MODEL=
# Batch Size
BATCH= 
# Trained model location
OUTPUT=
# Loss name (either BCE - DmyT - Triplet)
LOSS=
# Path to the dataset
DATASET=

python src/train.py --lr      $LR     \\
                    --epoch   $EPOCH  \\
                    --model   $MODEL  \\
                    --batch   $BATCH  \\
                    --output  $OUTPUT \\
                    --loss    $LOSS   \\
                    --dataset $DATASET
```

### Testing

Testing can be performed using the script `src/test.py`. You can use this bash script and complete the parameters

```bash
# Model name from TIMM library
MODEL=
# Location of the saved weights
WEIGHTS=
# Loss name (either BCE - DmyT - Triplet)
LOSS=
# Path to the dataset
DATASET=

python src/test.py  --model   $MODEl   \\
                    --weights $WEIGHTS \\
                    --loss    $LOSS    \\
                    --dataset $DATASET \\
```

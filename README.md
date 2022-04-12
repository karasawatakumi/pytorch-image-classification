# PyTorch Image Classification
Simple image classification for custom dataset (pytorch-lightning, timm).

Since it is implemented for easy handling, it may not be suitable for full-scale model building, but it has basic functionalities and is probably easy to modify.

## Data Preparation

Dataset preparation is simple. Prepare directories with the name of the class to train as follows, and store corresponding images in their directories.

```
./{dataset name}
├── train/
│   ├── {class1}/
│   ├── {class2}/
│   ├── ...
└── val
    ├── {class1}/
    ├── {class2}/
    ├── ...
```

### Sample scripts
For reference, I have prepared scripts to download `torchvision` datasets. `torchvision` originally provides us with datasets as `Dataset` class, but since the purpose of this repository is to run training for our own dataset, I save them once as jpeg images for easier understanding.

```
python scripts/download_and_generate_jpeg_dataset.py -d cifar10
```

```
usage: download_and_generate_jpeg_dataset.py [-h] --dataset_name DATASET_NAME
                                             [--outdir OUTDIR]

Script for generating dataset.

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME, -d DATASET_NAME
                        Dataset name to generate. (mnist or cifar10)
  --outdir OUTDIR, -o OUTDIR
                        Output directory. (default: dataset name)
```

```
./cifar10
├── raw/
│   ├── cifar-10-batches-py
│   └── cifar-10-python.tar.gz
├── train
│   ├── airplane
│   ├── automobile
│   ├── bird
│   ├── cat
│   ├── deer
│   ├── dog
│   ├── frog
│   ├── horse
│   ├── ship
│   └── truck
└── val
    ├── airplane
    ├── automobile
    ├── bird
    ├── cat
    ├── deer
    ├── dog
    ├── frog
    ├── horse
    ├── ship
    └── truck
```

- `raw/`: raw files downloaded by torchvision (Its content depends on dataset)


## Docker Environment

```
docker-compose build
docker-compose run --rm dev
```

Please see [docker-compose.yaml](./docker-compose.yaml), [Dockerfile](./Dockerfile), and [requirements.txt](./requirements.txt).

## Run

### Training
Simple implementation with everything in a single file (train.py).

```
python train.py -d cifar10
```

#### Detailed settings by command line ([code link](https://github.com/karasawatakumi/pytorch-image-classification/blob/main/train.py#L31-L42)):

```
usage: train.py [-h] --dataset DATASET [--outdir OUTDIR]
                [--model-name MODEL_NAME] [--img-size IMG_SIZE]
                [--epochs EPOCHS] [--save-interval SAVE_INTERVAL]
                [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS]
                [--gpu-ids GPU_IDS [GPU_IDS ...] | --n-gpu N_GPU]
                [--seed SEED]

Train classifier.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -d DATASET
                        Root directory of dataset
  --outdir OUTDIR, -o OUTDIR
                        Output directory
  --model-name MODEL_NAME, -m MODEL_NAME
                        Model name (timm)
  --img-size IMG_SIZE, -i IMG_SIZE
                        Input size of image
  --epochs EPOCHS, -e EPOCHS
                        Number of training epochs
  --save-interval SAVE_INTERVAL, -s SAVE_INTERVAL
                        Save interval (epoch)
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Batch size
  --num-workers NUM_WORKERS, -w NUM_WORKERS
                        Number of workers
  --gpu-ids GPU_IDS [GPU_IDS ...]
                        GPU IDs to use
  --n-gpu N_GPU         Number of GPUs
  --seed SEED           Seed
```



#### solver settings ([code link](https://github.com/karasawatakumi/pytorch-image-classification/blob/main/train.py#L18-L26)):

```python
OPT = 'adam'  # adam, sgd
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9  # only when OPT is sgd
BASE_LR = 0.001
LR_SCHEDULER = 'step'  # step, multistep, reduce_on_plateau
LR_DECAY_RATE = 0.1
LR_STEP_SIZE = 5  # only when LR_SCHEDULER is step
LR_STEP_MILESTONES = [10, 15]  # only when LR_SCHEDULER is multistep
```

#### augmentation settings ([code link](https://github.com/karasawatakumi/pytorch-image-classification/blob/main/train.py#L131-L145)):

```python
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
```

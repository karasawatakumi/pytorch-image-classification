# PyTorch Image Classification
Simple image classification for custom dataset based on [PyTorch Lightning](https://www.pytorchlightning.ai)  & [timm](https://github.com/rwightman/pytorch-image-models). You can train a classification model by simply preparing directories of images.

*I created a single-file (train.py), easily understood repository for my friend. So it may not be suitable for exhaustive experiments, but this has basic functionalities and is probably easy to use/modify.


## Docker Environment

```
docker-compose build
docker-compose run --rm dev
```

Or install directly with pip:

(\**The libraries are installed directly into your environment.*)

```
pip install -r docker/requirements.txt
```

Please see [docker-compose.yaml](./docker-compose.yaml), [Dockerfile](./docker/Dockerfile), and [requirements.txt](./docker/requirements.txt).


## Data Preparation

### Custom Dataset

Dataset preparation is simple. Prepare directories with the name of the class to train as follows, and store corresponding images in their directories. ([`ImageFolder`](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) class is used inside the loader.)

```
{dataset name}/
├── train/
│   ├── {class1}/
│   ├── {class2}/
│   ├── ...
└── val/
    ├── {class1}/
    ├── {class2}/
    ├── ...
```

### Sample Dataset
For reference, I have prepared a script to download `torchvision` datasets. 

`torchvision` originally provides us with datasets as `Dataset` class, but since the purpose of this repository is to run training for our own dataset, I save them once as jpeg images for easier understanding.

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

The script produces the following directory structure (when `outdir` is not specified):

```
cifar10/
├── raw/
│   ├── cifar-10-batches-py
│   └── cifar-10-python.tar.gz
├── train/
│   ├── airplane/
│   ├── automobile/
│   ├── bird/
│   ├── cat/
│   ├── deer/
│   ├── dog/
│   ├── frog/
│   ├── horse/
│   ├── ship/
│   └── truck/
└── val/
    ├── airplane/
    ├── automobile/
    ├── bird/
    ├── cat/
    ├── deer/
    ├── dog/
    ├── frog/
    ├── horse/
    ├── ship/
    └── truck/
```

- `raw/`: raw files downloaded by torchvision (Its content depends on dataset)


## Run

### Training
Simple implementation with everything in a single file ([train.py](./train.py))

Specify the dataset root directory containing the `train` and `val` directories.

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


#### solver settings ([code link](https://github.com/karasawatakumi/pytorch-image-classification/blob/main/train.py#L19-L26)):

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

#### transforms settings ([code link](https://github.com/karasawatakumi/pytorch-image-classification/blob/main/train.py#L105-L119)):

We use the [torchvision transforms](https://pytorch.org/vision/stable/transforms.html) because it is easy to use with the `ImageFolder` dataset.

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

#### tensorboard logging

We logged training with [tensorboard](https://pytorch.org/docs/stable/tensorboard.html) by default.

```
tensorboard --logdir ./results
```

![image](https://user-images.githubusercontent.com/13147636/163080755-e97695c3-80ed-4242-91e9-fd6b14b921a6.png)

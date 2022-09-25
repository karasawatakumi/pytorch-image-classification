import argparse
import os

from torchvision.datasets import CIFAR10, MNIST


def get_args():
    parser = argparse.ArgumentParser(description='Script for generating dataset.')
    parser.add_argument(
        '--dataset-name',
        '-d',
        required=True,
        help='Dataset name to generate. (mnist or cifar10)',
    )
    parser.add_argument(
        '--outdir', '-o', default=None, help='Output directory. (default: dataset name)'
    )
    args = parser.parse_args()
    return args


def main(dataset_name: str = 'mnist', outdir: str | None = None):
    # torchvision datasets
    if dataset_name == 'mnist':
        DATASET = MNIST
    elif dataset_name == 'cifar10':
        DATASET = CIFAR10
    else:
        raise NotImplementedError

    # outdir name
    if outdir is None:
        outdir = dataset_name

    for split in ['train', 'val']:
        download_dir = os.path.join(outdir, 'raw')
        dataset = DATASET(root=download_dir, train=(split == 'train'), download=True)
        classes = dataset.classes

        # output directories
        for class_name in classes:
            os.makedirs(os.path.join(outdir, split, class_name), exist_ok=True)

        # save images as jpeg
        for index, (image, target) in enumerate(dataset):
            image_name = f'image_{index}.jpg'
            class_name = classes[target]
            image_path = os.path.join(outdir, split, class_name, image_name)
            image.save(image_path)


if __name__ == '__main__':
    args = get_args()
    main(dataset_name=args.dataset_name, outdir=args.outdir)

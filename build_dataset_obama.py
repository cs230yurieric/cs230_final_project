"""Split the SIGNS dataset into train/dev/test and resize images to 64x64.

The SIGNS dataset comes in the following format:
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...

Original images have size (3024, 3024).
Resizing to (64, 64) reduces the dataset size from 1.16 GB to 4.7 MB, and loading smaller images
makes training faster.

We already have a test set created, so we only need to split "train_signs" into train and dev sets.
Because we don't have a lot of images and we want that the statistics on the dev set be as
representative as possible, we'll take 20% of "train_signs" as dev set.
"""

import argparse
import random
import os

from PIL import Image, ImageChops
from tqdm import tqdm

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

SIZE = 224

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/spectrograms', help="Directory with the spectrogram dataset")
parser.add_argument('--output_dir', default='data/altered_spectrograms', help="Where to write the new data")
parser.add_argument('--seed', default='1', help="Seed for random shuffling of files into train, dev, test sets")


def deborder_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = trim(image)

    #image = image.resize((size, size), Image.BILINEAR)
    image.load()

    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask = image.split()[3])

    background.save(os.path.join(output_dir, filename.split('/')[-1][:-3]) + "jpg", 'JPEG', quality = 100)
    #image.save(os.path.join(output_dir, filename.split('/')[-1]))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    # Get the filenames in each directory (train and test)

    data_dir = os.path.join(args.data_dir)
    filenames = os.listdir(data_dir)
    filenames = [os.path.join(data_dir, f) for f in filenames if f.endswith('.png')]

    # Split the images into 80% train, 10% dev, 10% test
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    #random.seed(230)
    random.seed(int(args.seed))
    filenames.sort()
    random.shuffle(filenames)

    split1 = int(0.8 * len(filenames))
    split2 = int(0.9 * len(filenames))

    train_filenames = filenames[:split1]
    dev_filenames = filenames[split1:split2]
    test_filenames = filenames[split2:]

    filenames = {'train': train_filenames,
                 'dev': dev_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}_spec'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            deborder_and_save(filename, output_dir_split, size=SIZE)

    print("Done building dataset")

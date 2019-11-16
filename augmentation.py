"""
Augment (double data set) using imgaug
Note:  can also take bounding boxes and transform

imgaug info:  https://github.com/aleju/imgaug
"""
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import argparse
import matplotlib.pyplot as plt
import os

def get_bboxes(annot_file):
    """Read file of annotations and create dictionary 
    with image as key"""
    image2bboxes = {}
    with open(annot_file) as f:
        for line in f:
            items = line.split()
            image = items[0]
            bboxes = items[1:]
            image2bboxes[image] = bboxes
    return image2bboxes

def main(image2bboxes, annot_file):
    bbs, images, classes, img_files = [], [], [], []
    for img_file in image2bboxes:
        bb = image2bboxes[img_file] # this has class, too
        images.append(plt.imread(os.path.join('..', img_file)))
        img_files.append(img_file)
        many_boxes = []
        many_classes = []
        # Parse out bboxes and classes, adding bboxes as imgaug BoundingBox objects
        for box in bb:
            box = [int(x) for x in box.split(',')]
            many_boxes.append(ia.BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
            many_classes.append(box[4])
        bbs.append(many_boxes)
        classes.append(many_classes)

    # To sometimes apply aug
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # The transformations!!!
    seq = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=0.05*255),
        # iaa.Affine(translate_px={"x": (1, 5)}),
        # sometimes(iaa.Fog()),
        sometimes(iaa.SigmoidContrast(cutoff=0.7)),
        # sometimes(iaa.Multiply(0.5)),
        sometimes(iaa.Add(-10)),
        sometimes(iaa.Pad(px=(256, 256, 0, 0)))
    ])

    images_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)

    # Save images
    for i in range(len(images)):
        img_file = img_files[i]
        # Create new file name and save
        name_spl = os.path.basename(img_file).split('.')
        ending = name_spl[-1]
        new_file_name = '.'.join(name_spl[0:-1]) + '_aug' + '.' + ending
        plt.imsave(os.path.join('..', 'data', 'JPEGImages', new_file_name), images_aug[i])

    # Save annotations to one file
    with open(annot_file.replace('.txt', '_aug.txt'), 'w') as f:
        for i in range(len(images)):
            items = []
            name_spl = os.path.basename(img_files[i]).split('.')
            ending = name_spl[-1]
            new_file_name = '.'.join(name_spl[0:-1]) + '_aug' + '.' + ending
            items.append('data/JPEGImages/' + new_file_name)
            many_boxes = bbs_aug[i]
            many_classes = classes[i]
            for j in range(len(many_boxes)):
                box = many_boxes[j]
                annot = ','.join([str(int(x)) for x in [box.x1, box.y1, box.x2, box.y2, many_classes[j]]])
                items.append(annot)
            f.write(' '.join(items) + '\n')
            
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--annot-path', type=str, dest='annot_file',
        help='Text file with images and their bounding boxes, \
            one image per line.'
    )
    args = parser.parse_args()
    # os.makedirs(os.path.join('..', 'data', 'test_aug'), exist_ok=True)
    image2bboxes = get_bboxes(args.annot_file)
    main(image2bboxes, args.annot_file)
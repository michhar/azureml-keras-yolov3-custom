"""
Quick ONNX inference test to ensure model loads and performs inference.
"""


from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import onnxruntime
import numpy as np
import argparse


def main(args):
    # image preprocessing
    img_path = args.image   # make sure the image is in img_path
    img_size = 416
    img = image.load_img(img_path, target_size=(img_size, img_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # runtime prediction
    sess = onnxruntime.InferenceSession(args.model_local)
    x = x if isinstance(x, list) else [x]
    feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
    pred_onnx = sess.run(None, feed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-local', type=str, dest='model_local',
        help='The path to the local ONNX model.'
    )
    parser.add_argument(
        '--image', type=str, dest='image',
        help='The path to a test image.'
    )

    args = parser.parse_args()
    main(args)
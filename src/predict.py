
import os
import glob
import json
import argparse
from utils.utils import calc_mean_score, save_json
from handlers.model_builder import Nima
from handlers.data_generator import TestDataGenerator
import time

BASE_MODEL_NAME = "MobileNet"
TECH_WEIGHTS_FILE = "/Users/marcomarchesi/Desktop/momento-ai/models/MobileNet/weights_mobilenet_technical_0.11.hdf5"
AESTHETIC_WEIGHTS_FILE = "/Users/marcomarchesi/Desktop/momento-ai/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5"


def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]

    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir, img_type='jpg'):
    img_paths = glob.glob(os.path.join(img_dir, '*.'+img_type))

    samples = []
    for img_path in img_paths:
        img_id = os.path.basename(img_path).split('.')[0]
        samples.append({'image_id': img_id})

    return samples


def predict(model, data_generator):
    return model.predict_generator(data_generator, workers=8, use_multiprocessing=True, verbose=1)


def main(image_source):

    start_time = time.time()

    img_format = 'jpg'
    
    # weights for aesthetic and technical analysis
    aw = 0.8
    tw = 1 - aw

    # load samples
    if os.path.isfile(image_source):
        image_dir, samples = image_file_to_json(image_source)
    else:
        image_dir = image_source
        samples = image_dir_to_json(image_dir, img_type=img_format)

    # build model and load weights
    nima = Nima(BASE_MODEL_NAME, weights=None)
    nima.build()

    # aesthetic analysis
    weights_file = AESTHETIC_WEIGHTS_FILE
    nima.nima_model.load_weights(weights_file)
    # initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, 64, 10, nima.preprocessing_function(),
                                       img_format=img_format)
    # get predictions
    aesthetic_predictions = predict(nima.nima_model, data_generator)

    # technical analysis
    weights_file = TECH_WEIGHTS_FILE
    nima.nima_model.load_weights(weights_file)
    # initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, 64, 10, nima.preprocessing_function(),
                                       img_format=img_format)
    # get predictions
    technical_predictions = predict(nima.nima_model, data_generator)
    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        sample['aesthetic'] = calc_mean_score(aesthetic_predictions[i])
        sample['technical'] = calc_mean_score(technical_predictions[i])
        sample['aesthetic+technical'] = tw * calc_mean_score(technical_predictions[i]) + aw * calc_mean_score(aesthetic_predictions[i])

    # order by total score
    sorted_samples = sorted(samples, key=lambda k:k['aesthetic+technical'], reverse=True)
    print(json.dumps(sorted_samples, indent=2))

    print ("time: %fs" % (time.time() - start_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-is', '--image-source', help='image directory or file', required=True)

    args = parser.parse_args()

    # run
    main(args.image_source)

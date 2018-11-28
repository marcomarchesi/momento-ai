
import os
import glob
import json
import argparse
from utils.utils import calc_mean_score, save_json, load_image
from handlers.model_builder import Nima
from handlers.data_generator import TestDataGenerator
import time
import numpy as np
import random
from inception_score import get_inception_score


BASE_MODEL_NAME = "MobileNet"
TECH_WEIGHTS_FILE = "/Users/marcomarchesi/Desktop/momento-ai/models/MobileNet/weights_mobilenet_technical_0.11.hdf5"
AESTHETIC_WEIGHTS_FILE = "/Users/marcomarchesi/Desktop/momento-ai/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5"

# weights for aesthetic and technical analysis
AW = 0.5
TW = 1 - AW

# A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
IS_IMAGE_SIZE = 224

def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]

    return img_dir, [{'image_id': img_id}]

def image_dir_to_json(img_dir, img_type='jpg', img_size=IS_IMAGE_SIZE):
    img_paths = glob.glob(os.path.join(img_dir, '*.'+ img_type))

    samples = []
    
    for img_path in img_paths:
        # print(os.path.basename(img_path)[:-4])
        # img_id = os.path.basename(img_path).split('.')[0]
        img_id = os.path.basename(img_path)[:-4]
        samples.append({'image_id': img_id})

    return samples

def images_to_np(images, img_type='jpg', img_size=IS_IMAGE_SIZE):
    np_images = np.empty((len(images), 3, img_size, img_size), dtype=np.uint8)
    for i, img_path in enumerate(images):
        img = load_image(img_path, (img_size,img_size))
        img = np.reshape(img, (3,img_size,img_size))
        np_images[i] = img
    return np_images
    

def select_images(img_dir, samples, fraction=3):
    # select the best images:
    n = len(samples) // fraction
    if n > 20:
        n = 20
    
    return samples[:n], samples[-5:]

def predict(model, data_generator):
    return model.predict_generator(data_generator, workers=8, use_multiprocessing=True, verbose=1)


def main(image_source):

    start_time = time.time()

    img_format = 'jpg'
    

    # load samples
    if os.path.isfile(image_source):
        image_dir, samples = image_file_to_json(image_source)
    else:
        image_dir = image_source
        samples = image_dir_to_json(image_dir, img_type=img_format)


    # build model and load weights
    nima = Nima(BASE_MODEL_NAME, weights=None)
    nima.build()


    # initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, 64, 10, nima.preprocessing_function(),
                                       img_format=img_format)

    # aesthetic analysis
    weights_file = AESTHETIC_WEIGHTS_FILE
    nima.nima_model.load_weights(weights_file)
    # get predictions
    aesthetic_predictions = predict(nima.nima_model, data_generator)
    # technical analysis
    weights_file = TECH_WEIGHTS_FILE
    nima.nima_model.load_weights(weights_file)
    # get predictions
    technical_predictions = predict(nima.nima_model, data_generator)

    print("Processing %i images" % len(samples))


    max_t = 0
    max_a = 0
    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        sample['aesthetic'] = calc_mean_score(aesthetic_predictions[i])
        max_a = max(sample['aesthetic'], max_a)
        sample['technical'] = calc_mean_score(technical_predictions[i])
        max_t = max(sample['technical'], max_t)
    for sample in samples:
        sample['aesthetic+technical'] = TW * (sample['technical'] / max_t) + AW * (sample['aesthetic'] / max_a)
    # order by total score
    sorted_samples = sorted(samples, key=lambda k:k['aesthetic+technical'], reverse=True)


    # select the best N images with the highest score
    best_samples, worst_samples = select_images(image_source, sorted_samples)

    image_paths = []
    for sample in best_samples:
        img_path = os.path.join(image_source, sample["image_id"]) + ".jpg"
        image_paths.append(img_path)

    # Inception Score Optimization
    # Select a random subset of images + the highest scored one and calculate the IS

    print("Best Image is:")
    print(json.dumps(best_samples[0], indent=2))

    best_variety = []
    inception_score = 0
    for i in range(5):
        random_selected_images = random.sample(image_paths, 5)
        if image_paths[0] not in random_selected_images:
            del random_selected_images[-1]
            random_selected_images.append(image_paths[0])
            
        _is, _ = get_inception_score(images_to_np(random_selected_images))
        # print(_is)
        if _is > inception_score:
            best_variety = random_selected_images
            inception_score = _is
    # print("Best Inception Score is %f" % inception_score)

    print("And the Best Variety is:")
    print(json.dumps(best_variety, indent=2))

    print("Worst Images were:")
    print(json.dumps(worst_samples[4], indent=2))
    print(json.dumps(worst_samples[3], indent=2))
    print(json.dumps(worst_samples[2], indent=2))
    print(json.dumps(worst_samples[1], indent=2))
    print(json.dumps(worst_samples[0], indent=2))

    print ("Total Processing Time: %fs" % (time.time() - start_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-is', '--image-source', help='image directory or file', required=True)

    args = parser.parse_args()

    # run
    main(args.image_source)

# for inception score
import tensorflow as tf
import functools
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
tfgan = tf.contrib.gan
import numpy as np
import time
from keras.applications.imagenet_utils import decode_predictions

BATCH_SIZE = 1

session = tf.InteractiveSession()

# Run images through Inception.
inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
def inception_logits(images = inception_images, num_splits = 1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits = num_splits)
    logits = functional_ops.map_fn(
        fn = functools.partial(tfgan.eval.run_inception, output_tensor = 'logits:0'),
        elems = array_ops.stack(generated_images_list),
        parallel_iterations = 1,
        back_prop = False,
        swap_memory = True,
        name = 'RunClassifier')
    logits = array_ops.concat(array_ops.unstack(logits), 0)
    return logits

logits=inception_logits()

def get_inception_probs(inps):
    
    n_batches = len(inps) // BATCH_SIZE
    preds = np.zeros([n_batches * BATCH_SIZE, 1000], dtype = np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] / 255. * 2 - 1
        preds[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = logits.eval({inception_images:inp})[:, :1000]
        print(decode_predictions(preds, top=3))
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
    
    return preds

def preds2score(preds, splits=10):
    # print("NEW SCORE")
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]  # (5,1000)
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))   # (5,1000)
        kl = np.mean(np.sum(kl, 1)) #scalar value
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def get_inception_score(images, splits=1):
    assert(type(images) == np.ndarray)
    assert(len(images.shape) == 4)
    assert(images.shape[1] == 3)
    assert(np.min(images[0]) >= 0 and np.max(images[0]) > 10), 'Image values should be in the range [0, 255]'
    # print('Calculating Inception Score with %i images in %i splits' % (images.shape[0], splits))
    start_time=time.time()
    preds = get_inception_probs(images)  #each preds[i] has size=1000 (categories)  preds.shape=(5,1000)
    print(preds)
    print(images)
    # for pred in preds:
    #     pred = pred.tolist()
    #     print(pred.index(max(pred)))


    mean, std = preds2score(preds, splits)
    # print('Inception Score calculation time: %f s' % (time.time() - start_time))
    return mean, std  # Reference values: 11.34 for 49984 CIFAR-10 training set images, or mean=11.31, std=0.08 if in 10 splits.
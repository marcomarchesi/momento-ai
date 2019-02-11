# Model Converter -- From Keras to CoreML

import coremltools
import keras
from keras.models import load_model
# from argparse import ArgumentParser

from keras.utils.generic_utils import CustomObjectScope
from keras import applications
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.nn import relu6
# from keras_applications.mobilenet_v2 import DepthwiseConv2D



# parser = ArgumentParser()
# parser.add_argument("--model", type=str)
# parser.add_argument("--mlmodel", type=str)

# args = parser.parse_args()


# with CustomObjectScope({'relu6': keras.layers.activations.relu, 'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D }):
#     model = load_model(args.model)
#     coreml_model = coremltools.converters.keras.convert(model, input_names = 'input_1', image_input_names='input_1')
#     coreml_model.save(args.mlmodel)

# # with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6}):
# #     model = load_model(args.model)
# #     coreml_model = coremltools.converters.keras.convert(model, input_names = 'input_1', image_input_names='input_1')
# #     coreml_model.save(args.mlmodel)
with CustomObjectScope({'relu6': relu6}):
    aesthetic_model = load_model('aesthetic.h5')
    coreml_aesthetic_model = coremltools.converters.keras.convert(aesthetic_model, input_names = 'input_1', image_input_names='input_1')
    coreml_aesthetic_model.save('aesthetic.mlmodel')

    technical_model = load_model('technical.h5')
    coreml_technical_model = coremltools.converters.keras.convert(technical_model, input_names = 'input_1', image_input_names='input_1')
    coreml_technical_model.save('technical.mlmodel')


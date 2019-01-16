# Model Converter -- From Keras to CoreML

import coremltools
from keras.models import load_model


aesthetic_model = load_model('nima_aesthetic.h5')
coreml_aesthetic_model = coremltools.converters.keras.convert(aesthetic_model, input_names = 'input_1', image_input_names='input_1')
coreml_aesthetic_model.save('nima_aesthetic.mlmodel')

technical_model = load_model('nima_technical.h5')
coreml_technical_model = coremltools.converters.keras.convert(technical_model, input_names = 'input_1', image_input_names='input_1')
coreml_technical_model.save('nima_technical.mlmodel')


import os
import glob
import time
import argparse

# Modified file compared to Alhashim et al.
# Original code can be found at https://github.com/ialhashim/DenseDepth/blob/master/evaluate.py

# This script evaluates a given model on the test set and displays errors + time needed

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from loss import depth_loss_function, ScaledSIError, root_mean_squared_error
from utils import predict, load_images, display_images, evaluate
from matplotlib import pyplot as plt
from model import EffNetFullyFledged

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'depth_loss_function': ScaledSIError, 'EffNetFullyFledged': EffNetFullyFledged}

# Load model into GPU / CPU
print('Loading model...')
model = load_model(args.model, custom_objects=custom_objects, compile=False)

# Load test data
print('Loading test data...', end='')
import numpy as np
from data import extract_zip
data = extract_zip('nyu_test.zip') # adjust file path if needed
from io import BytesIO
rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
depth = np.load(BytesIO(data['eigen_test_depth.npy']))
crop = np.load(BytesIO(data['eigen_test_crop.npy']))
print('Test data loaded.\n')

start = time.time()
print('Testing...')

e = evaluate(model, rgb, depth, crop, batch_size=6)

print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10', 'ord'))
print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5], e[6]))

end = time.time()
print('\nTest time', end-start, 's')

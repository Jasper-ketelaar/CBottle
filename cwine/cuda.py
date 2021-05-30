import tensorflow as tf
from tensorflow import config
print(config.experimental.get_visible_devices())
print(config.get_visible_devices())
print(tf.__version__)
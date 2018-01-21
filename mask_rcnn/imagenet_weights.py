from keras.utils.data_utils import get_file


def get_imagenet_weights():
  """Downloads ImageNet trained weights from Keras.
  Returns path to weights file.
  """
  tf_weights_path_no_top = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

  return get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                  tf_weights_path_no_top,
                  cache_subdir='models',
                  md5_hash='a268eb855778b3df3c7506639542a6af')

import numpy as np

############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
  """Takes attributes of an image and puts them in one 1D array. Use
  parse_image_meta() to parse the values back.

  image_id: An int ID of the image. Useful for debugging.
  image_shape: [height, width, channels]
  window: (y1, x1, y2, x2) in pixels. The area of the image where the real
          image is (excluding the padding)
  active_class_ids: List of class_ids available in the dataset from which
      the image came. Useful if training on images from multiple datasets
      where not all classes are present in all datasets.
  """
  meta = np.array(
    [image_id] +            # size=1
    list(image_shape) +     # size=3
    list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
    list(active_class_ids)  # size=num_classes
  )
  return meta


def mold_image(images, config):
  """Takes RGB images with 0-255 values and subtraces
  the mean pixel and converts it to float. Expects image
  colors in RGB order.
  """
  return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
  """Takes a image normalized with mold() and returns the original."""
  return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)

def parse_image_meta_graph(meta):
  """Parses a tensor that contains image attributes to its components.
  See compose_image_meta() for more details.

  meta: [batch, meta length] where meta length depends on NUM_CLASSES
  """
  image_id = meta[:, 0]
  image_shape = meta[:, 1:4]
  window = meta[:, 4:8]
  active_class_ids = meta[:, 8:]
  return [image_id, image_shape, window, active_class_ids]


# Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
  """Parses an image info Numpy array to its components.
  See compose_image_meta() for more details.
  """
  image_id = meta[:, 0]
  image_shape = meta[:, 1:4]
  window = meta[:, 4:8]   # (y1, x1, y2, x2) window of image in in pixels
  active_class_ids = meta[:, 8:]
  return image_id, image_shape, window, active_class_ids
import tensorflow as tf


def trim_zeros_graph(boxes, name=None):
  """Often boxes are represented with matricies of shape [N, 4] and
  are padded with zeros. This removes zero boxes.

  boxes: [N, 4] matrix of boxes.
  non_zeros: [N] a 1D boolean mask identifying the rows to keep
  """
  non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
  boxes = tf.boolean_mask(boxes, non_zeros, name=name)
  return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
  """Picks different number of values from each row
  in x depending on the values in counts.
  """
  outputs = []
  for i in range(num_rows):
    outputs.append(x[i, :counts[i]])
  return tf.concat(outputs, axis=0)

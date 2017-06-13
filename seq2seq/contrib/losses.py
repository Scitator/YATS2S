import tensorflow as tf


def contrastive_loss(labels, distances, margin=1.0):
    """
    ## y = 0 - same class, y = 1 - other, No
    """
    # between_class = tf.multiply(1.0 - labels, tf.square(distances))
    between_class = tf.multiply(labels, tf.square(distances))
    max_part = tf.square(tf.maximum(margin - distances, 0))
    # within_class = tf.multiply(labels, max_part)
    within_class = tf.multiply(1.0 - labels, max_part)
    return 0.5 * tf.reduce_mean(within_class + between_class)

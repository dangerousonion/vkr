import tensorflow as tf


def SoftmaxLoss():
    """softmax loss"""
    def softmax_loss(y_true, y_pred):
        # y_true: sparse target  (None, 1)      None stands for a batch size
        # y_pred: logist         (None, 54)
        print('y_true:', y_true.shape, ' y_pred:', y_pred.shape);

        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32) # (None,)
        print('y_true:', y_true.shape)

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred) # (None,)
        return tf.reduce_mean(ce) # just the mean
    return softmax_loss

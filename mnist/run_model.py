import argparse

import tensorflow as tf
import numpy as np

import json, socket, os
from model.mnist import mnist_model

tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features, labels, mode):

    input_tensor = tf.placeholder_with_default(features['x'], shape=[None,28,28,1],name="input")

    with tf.variable_scope("model"):
        logits = mnist_model(input_tensor)

    labels_tensor = tf.placeholder_with_default(labels, shape=[None],name="labels")

    predictions = {
        "classes": tf.cast(tf.argmax(input=logits, axis=1),tf.int32),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_tensor, logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions["classes"], labels_tensor), tf.float32), name="accuracy")

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels_tensor, predictions=predictions["classes"])
        }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train an mnist model with a trojan')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Number of images in batch.')
    # parser.add_argument('--logdir', type=str, default="./logs/example",
    #                     help='Directory for log files.')
    parser.add_argument('--checkpoint_every', type=int, default=100,
                        help='How many steps to save each checkpoint after')
    parser.add_argument('--num_steps', type=int, default=10000,
                        help='Number of training steps.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for training.')
    parser.add_argument('--dropout_rate', type=float, default=0.4,
                        help='Dropout keep probability.')
    args = parser.parse_args()


    with open('config_mnist.json') as config_file:
        config = json.load(config_file)

    if socket.gethostname() == 'deep':
        logdir = config['logdir_deep']

    # Load training and test data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    test_data = mnist.test.images
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    train_data = train_data.reshape([-1,28,28,1])
    test_data = test_data.reshape([-1,28,28,1])

    mnist_classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=os.path.join(logdir, 'pretrained'))

    tensors_to_log = {"accuracy": "accuracy"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=args.batch_size,
        num_epochs=None,
        shuffle=True)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=args.num_steps,
        hooks=[logging_hook])

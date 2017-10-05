import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from tensorflow.python.ops import init_ops
from click.core import batch


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights

    vgg_tag = 'vgg16'

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    graph = tf.get_default_graph()

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    conv0_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    deconv1_output = tf.layers.conv2d_transpose(conv0_1x1, num_classes, 4, 2, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    conv1_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    deconv1_output = tf.add(deconv1_output, conv1_1x1)
    
    deconv2_output = tf.layers.conv2d_transpose(deconv1_output, num_classes, 4, 2, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    conv2_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    deconv3_output = tf.add(deconv2_output, conv2_1x1)
    
    output = tf.layers.conv2d_transpose(deconv3_output, num_classes, 16, strides=(8, 8), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
#     print_tensor = tf.Print(conv_1x1, [tf.shape(conv_1x1)[1:]])
    return output #, print_tensor

tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)

def mean_iou(correct_label, nn_last_layer, num_classes):
    """
    Build the TensorFLow mean IOU and its operation.
    :param nn_last_layer:TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param num_classes: Number of classes to classify
    :return: Tuple of (iou, iou_op)
    """
    # TODO: Use `tf.metrics.mean_iou` to compute the mean IoU.
    labels = tf.reshape(correct_label, (-1, num_classes))
    labels = tf.cast(labels,tf.int32)
#     label_locations = tf.where(tf.equal(labels, 1))
#     dense_labels = label_locations[:,1]
    dense_labels = tf.argmax(labels, axis=1)
    
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    softmax_logits = tf.nn.softmax(logits)
    predictions = tf.argmax(softmax_logits, axis=1)

#     pred_locations = tf.where(tf.greater(softmax_logits, 0.5))
#     predictions = pred_locations[:,1]
#     predictions = tf.one_hot(tf.nn.top_k(softmax_logits).indices, tf.shape(im_softmax)[0])


    iou, iou_op = tf.metrics.mean_iou(dense_labels, predictions, num_classes)
    
    return iou, iou_op

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, iou, iou_op, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    for epoch in range(epochs):
        print(epoch)
        for image, label in get_batches_fn(batch_size):
            _, _, loss, IOU = sess.run([train_op, iou_op, cross_entropy_loss, iou],feed_dict = {input_image: image, correct_label: label, keep_prob: 0.8, learning_rate: 1e-3})
            print(loss, " : ", IOU)
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
#     learning_rate = 1e-4 # based on FCN-8 paper
    batch_size = 20 # FCN-8 paper
    epochs = 1
    correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
    learning_rate_pl = tf.placeholder(tf.float32)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate_pl, num_classes)
        iou, iou_op = mean_iou(correct_label, nn_last_layer, num_classes)
        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        # need to initialize local variables for this to run `tf.metrics.mean_iou`
        sess.run(tf.local_variables_initializer())
        # TODO: use `mean_iou` to compute the mean IoU

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, iou, iou_op, input_image,
             correct_label, keep_prob, learning_rate_pl)
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()

import os.path
import tensorflow as tf
import helper
import warnings
import shutil
import time

from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
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
    with tf.name_scope('deconv'):
        
        regularizer = tf.contrib.layers.l2_regularizer(1e-3)
        initializer = tf.contrib.layers.xavier_initializer()
        
        conv0_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), padding='same', \
                                     kernel_regularizer=regularizer, kernel_initializer=initializer, \
                                     name='layer7_to_conv1x1')
        deconv1_output = tf.layers.conv2d_transpose(conv0_1x1, num_classes, 4, 2, padding='same', \
                                                    kernel_regularizer=regularizer, \
                                                    kernel_initializer=initializer, name='deconv_layer_1')
        
        conv1_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1), padding='same', \
                                     kernel_regularizer=regularizer, kernel_initializer=initializer, \
                                     name='layer4_to_conv1x1')
        deconv1_output = tf.add(deconv1_output, conv1_1x1, name='skip_connection_1')
        deconv2_output = tf.layers.conv2d_transpose(deconv1_output, num_classes, 4, 2, padding='same', \
                                                    kernel_regularizer=regularizer, \
                                                    kernel_initializer=initializer, name='deconv_layer_2')
        
        conv2_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1), padding='same', \
                                     kernel_regularizer=regularizer, kernel_initializer=initializer, \
                                     name='layer3_to_conv1x1')
        deconv3_output = tf.add(deconv2_output, conv2_1x1, name='skip_connection_2')
        output = tf.layers.conv2d_transpose(deconv3_output, num_classes, 16, strides=(8, 8), padding='same', \
                                            kernel_regularizer=regularizer, kernel_initializer=initializer, \
                                            name='deconv_layer_3')
    
    return output

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
    logits = tf.reshape(nn_last_layer, (-1, num_classes),name='logits')
    cross_entropy_loss = \
    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label, name='Xentropy'), \
                   name='cross_entropy_loss')
    
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss,global_step=global_step)

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
    dense_labels = tf.argmax(labels, axis=1)
    
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    softmax_logits = tf.nn.softmax(logits)
    predictions = tf.argmax(softmax_logits, axis=1)

    iou, iou_op = tf.metrics.mean_iou(dense_labels, predictions, num_classes, name='mean_iou')
    
    return iou, iou_op

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss, iou, iou_op, input_image,
             correct_label, keep_prob, learning_rate, summary, summary_writer, saver, output_dir):
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
        
        lr = 1e-04
        kp = 0.8
        
        for image, label in get_batches_fn(batch_size):
            _, _, cost, IOU = sess.run([train_op, iou_op, loss, iou], \
                                       feed_dict = {input_image: image, correct_label: label, \
                                                    keep_prob: kp, learning_rate: lr})
        print('Epoch: {:<4} - Cost: {:<8.3} IOU: %{:<5.3}'.format(epoch, cost, IOU*100))
        summary_str = sess.run(summary, \
                               feed_dict = {input_image: image, correct_label: label, \
                                            keep_prob: kp, learning_rate: lr})
        summary_writer.add_summary(summary_str, epoch)
        summary_writer.flush()
        # Save a checkpoint and evaluate the model periodically.
        checkpoint_file = os.path.join(output_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=epoch)
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
#     learning_rate = 1e-4 # based on FCN-8 paper
    batch_size = 4 # FCN-8 paper
    epochs = 50
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
        # OPTINAL: load a half-trained model and fine-tune it
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
                
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate_pl, num_classes)
        #the output of inference
        softmax_logits = tf.nn.softmax(logits, name='softmax')

        
        # add regularization terms to loss
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.reduce_sum(reg_variables)
        loss = cross_entropy_loss + reg_term
        
        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar('loss', loss)

        # TODO: use `mean_iou` to compute the mean IoU
        iou, iou_op = mean_iou(correct_label, nn_last_layer, num_classes)
        tf.summary.scalar('IOU', iou)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()
        
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        
        # Make folder for current run
        output_dir = os.path.join(runs_dir, str(time.time()))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(output_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        # need to initialize local variables for this to run `tf.metrics.mean_iou`
        sess.run(tf.local_variables_initializer())

        tf.train.write_graph(sess.graph, output_dir, 'base_graph.pb',as_text=False)
        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss, iou, iou_op, input_image,
            correct_label, keep_prob, learning_rate_pl, summary, summary_writer, saver, output_dir)
        
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(output_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video
        # Check the inference-walkthrough for video_inference.py

if __name__ == '__main__':
    run()

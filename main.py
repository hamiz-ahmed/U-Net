"""
Copyright 2017, University of Freiburg.
Muhammad Hamiz Ahmed <hamizahmed93@gmail.com>

This is the code for Deep Learning Lab Exercise 4
"""



import h5py
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt


class Data:
    def __init__(self):
        with h5py.File("cell_data.h5", "r") as data:
            self.train_images = [data["/train_image_{}".format(i)][:] for i in range(28)]
            self.train_labels = [data["/train_label_{}".format(i)][:] for i in range(28)]
            self.test_images = [data["/test_image_{}".format(i)][:] for i in range(3)]
            self.test_labels = [data["/test_label_{}".format(i)][:] for i in range(3)]

        self.input_resolution = 300
        self.label_resolution = 116

        self.offset = (300 - 116) // 2

    def get_train_image_list_and_label_list(self):
        n = random.randint(0, len(self.train_images) - 1)
        x = random.randint(0, (self.train_images[n].shape)[1] - self.input_resolution - 1)
        y = random.randint(0, (self.train_images[n].shape)[0] - self.input_resolution - 1)
        image = self.train_images[n][y:y + self.input_resolution, x:x + self.input_resolution, :]

        x += self.offset
        y += self.offset
        label = self.train_labels[n][y:y + self.label_resolution, x:x + self.label_resolution]

        return [image], [label]

    def get_test_image_list_and_label_list(self):
        coord_list = [[0, 0], [0, 116], [0, 232],
                      [116, 0], [116, 116], [116, 232],
                      [219, 0], [219, 116], [219, 232]]

        image_list = []
        label_list = []

        for image_id in range(3):
            for y, x in coord_list:
                image = self.test_images[image_id][y:y + self.input_resolution, x:x + self.input_resolution, :]
                image_list.append(image)
                x += self.offset
                y += self.offset
                label = self.test_labels[image_id][y:y + self.label_resolution, x:x + self.label_resolution]
                label_list.append(label)

        return image_list, label_list


class Layers:
    def __init__(self):
        self.learning_rate = tf.placeholder(tf.float32)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        # stride [1, x_movement, y_movement, 1]
        # Must have strides[0] = strides[3] = 1
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def max_pool_2x2(self, x):
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    def conv2d_transpose(self, x, output_filter_size):
        return tf.layers.conv2d_transpose(inputs=x, filters=output_filter_size,
                                          kernel_size=2,
                                          strides=2,
                                          padding='VALID')

    def create_convolution_layer(self, input_image, num_filters,
                                 filter_size=3):
        # creates a convolution layer by default
        num_input_channels = int(input_image.get_shape()[3])
        weight_shape = [filter_size, filter_size, num_input_channels, num_filters]
        weights = self.weight_variable(shape=weight_shape)
        biases = self.bias_variable([num_filters])

        layer = self.conv2d(input_image, weights)
        layer = tf.nn.relu(layer + biases)

        return layer

    def do_pooling(self, layer):
        return self.max_pool_2x2(layer)

    def create_transposed_convolution(self, input_image):
        output_filters = int(int(input_image.get_shape()[3])/2)
        return self.conv2d_transpose(input_image, output_filters)

    def crop_tensor(self, tensor, height, width):
        return tf.image.resize_image_with_crop_or_pad(tensor, height, width)

    def crop_and_merge_layers(self, layer_to_be_cropped, second_layer):
        size = int(second_layer.get_shape()[1])
        cropped_tensor_1 = nn.crop_tensor(layer_to_be_cropped, size, size)
        return tf.concat([second_layer, cropped_tensor_1], 3)


class Utilities:

    def plot_image(self, im):
        figure = plt.figure()
        ax = plt.Axes(figure, [0., 0., 1., 1.])
        figure.add_axes(ax)
        ax.imshow(im, cmap='gray')
        plt.show()

    def get_accuracy(self, prediction, labels):
        prediction = np.argmax(prediction, axis=3)

        correct_pix = np.sum(prediction == labels)
        inccorect_pix = np.sum(prediction != labels)
        total_pixels = correct_pix + inccorect_pix

        accuracy = correct_pix / (total_pixels + inccorect_pix)
        return accuracy

    def compute_validation_accuracy(self, sess, validation_xs, validation_ys):
        y_pre = sess.run(output_conv, feed_dict={x_image: validation_xs, y_label: validation_ys})
        return self.get_accuracy(y_pre, validation_ys)

    def compute_training_accuracy(self, nn_output, y_labels):
        return self.get_accuracy(nn_output, y_labels)

    def write_to_file(self, file_name, value):
        file_name += '.txt'
        with open(file_name, 'a') as the_file:
            the_file.write(value + '\n')


def create_unet_architecture(nn:Layers):
    # Extract Images
    global x_image, y_label, output_conv

    x_image = tf.placeholder(tf.float32, [None, 300, 300, 1])
    y_label = tf.placeholder(tf.int32, [None, 116, 116])

    # 2 convolutions and pooling
    conv_layer1 = nn.create_convolution_layer(x_image, num_filters=32)  # returns 298x298x32
    conv_layer2 = nn.create_convolution_layer(conv_layer1, num_filters=32)
    pooling_layer1 = nn.do_pooling(conv_layer2)  # returns 148x148x32

    # 2 convolutions and pooling
    conv_layer3 = nn.create_convolution_layer(pooling_layer1, num_filters=64)
    conv_layer4 = nn.create_convolution_layer(conv_layer3, num_filters=64)
    pooling_layer2 = nn.do_pooling(conv_layer4)  # returns 72x72x64

    # 2 convolutions and pooling
    conv_layer5 = nn.create_convolution_layer(pooling_layer2, num_filters=128)
    conv_layer6 = nn.create_convolution_layer(conv_layer5, num_filters=128)
    pooling_layer3 = nn.do_pooling(conv_layer6)  # returns 34x34x128

    # 2 convolutions and pooling
    conv_layer7 = nn.create_convolution_layer(pooling_layer3, num_filters=256)
    conv_layer8 = nn.create_convolution_layer(conv_layer7, num_filters=256)  # 30x30x256
    pooling_layer4 = nn.do_pooling(conv_layer8)  # returns 15x15x256

    # 2 convolutions and up conv
    conv_layer9 = nn.create_convolution_layer(pooling_layer4, num_filters=512)
    conv_layer10 = nn.create_convolution_layer(conv_layer9, num_filters=512)
    up_conv_layer1 = nn.create_transposed_convolution(conv_layer10)  # returns 22x22x256

    # merge conv layer 8 and up conv layer 1, shape = 22,22,512
    merged_1 = nn.crop_and_merge_layers(layer_to_be_cropped=conv_layer8, second_layer=up_conv_layer1)

    # 2 convolutions and up conv
    conv_layer11 = nn.create_convolution_layer(merged_1, num_filters=512)
    conv_layer12 = nn.create_convolution_layer(conv_layer11, num_filters=256)
    up_conv_layer2 = nn.create_transposed_convolution(conv_layer12)  # returns 36x36x128

    # merge conv layer 6 and up conv layer 2, shape = 36,36,256
    merged_2 = nn.crop_and_merge_layers(layer_to_be_cropped=conv_layer6, second_layer=up_conv_layer2)

    # 2 convolutions and up conv
    conv_layer13 = nn.create_convolution_layer(merged_2, num_filters=256)
    conv_layer14 = nn.create_convolution_layer(conv_layer13, num_filters=128)
    up_conv_layer3 = nn.create_transposed_convolution(conv_layer14)  # returns 64x64x64

    # merge conv layer 4 and up conv layer 3, shape = 64,64,128
    merged_3 = nn.crop_and_merge_layers(layer_to_be_cropped=conv_layer4, second_layer=up_conv_layer3)

    # 2 convolutions and up conv
    conv_layer15 = nn.create_convolution_layer(merged_3, num_filters=128)
    conv_layer16 = nn.create_convolution_layer(conv_layer15, num_filters=64)
    up_conv_layer4 = nn.create_transposed_convolution(conv_layer16)  # returns 120*120*32

    # merge conv layer 2 and up conv layer 4, shape = 120,120,64
    merged_4 = nn.crop_and_merge_layers(layer_to_be_cropped=conv_layer2, second_layer=up_conv_layer4)

    # 2 convolutions and output
    conv_layer17 = nn.create_convolution_layer(merged_4, num_filters=64)
    conv_layer18 = nn.create_convolution_layer(conv_layer17, num_filters=32)
    output_conv = nn.create_convolution_layer(conv_layer18, num_filters=2, filter_size=1)


def perform_tf_operations():
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    y_plot_validation = []
    y_plot_train = []
    x_plot = []

    for i in range(40000):
        batch_xs, batch_ys = data.get_train_image_list_and_label_list()
        output, _ = sess.run([output_conv, train_step], feed_dict={x_image: batch_xs, y_label: batch_ys})

        if i % 100 == 0:
            print("Step: ", i)
            training_accuracy = utils.compute_training_accuracy(output, batch_ys)
            valid_images, valid_labels = data.get_test_image_list_and_label_list()
            validation_accuracy = utils.compute_validation_accuracy(sess, valid_images, valid_labels)
            print("Validation accuracy: ", validation_accuracy)

            utils.write_to_file('epochs', str(i))
            x_plot.append(i)

            utils.write_to_file('validation_accuracy', str(validation_accuracy))
            y_plot_validation.append(validation_accuracy)

            utils.write_to_file('training_Accuracy', str(training_accuracy))
            y_plot_train.append(training_accuracy)

    plt.plot(x_plot, y_plot_validation, label='validation')
    plt.plot(x_plot, y_plot_train, label='training')


    plt.xlabel('steps')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.show()

    show_segmented_images(sess)
    sess.close()


def show_segmented_images(sess):
    test_im, _ = data.get_test_image_list_and_label_list()

    for i in range(2):
        # plot original images
        image = test_im[i]
        im = np.array([[p[0] for p in l] for l in image])
        utils.plot_image(im)
        # plot segmented images
        test_out = sess.run(output_conv, feed_dict={x_image: test_im})
        test_prediction = np.argmax(test_out, axis=3)
        utils.plot_image(test_prediction[i])

    
if __name__ == '__main__':
    data = Data()
    nn = Layers()
    utils = Utilities()

    create_unet_architecture(nn)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label, logits=output_conv)
    total_loss = tf.reduce_mean(loss)
    train_step = tf.train.AdamOptimizer(0.0001, 0.95, 0.99).minimize(total_loss)

    perform_tf_operations()








'''ncGAN implementation: cGAN but input of generator is enhanced
with high-order features that discriminator doesn't observe'''

import tensorflow as tf

import os
import sys
import time

import matplotlib.pyplot as plt
try:
    from IPython.display import clear_output
except ModuleNotFoundError:
    pass

import numpy as np
import cv2

BATCH_SIZE = 1
BUFFER_SIZE = 400
IMG_WIDTH = 256
IMG_HEIGHT = 256

### preprocessing funs

def enhance(satl, thresl=50, thresh=150):
    '''Add detected edges as 4th channel of numpy.ndarray. `satl` should
    be an image in range [0,255]. Threshold dictate the specifics of the
    Canny edge detector operations'''
    # canny requires integer input
    edges = cv2.Canny(satl.astype(np.uint8), thresl, thresh, L2gradient=True)
    ninput = np.dstack((satl, edges))
    return ninput

def tf_enhance(satl, thresl=50, thresh=150):
    '''Add detected edges as 4th channel of tf.Tensor. `satl` should be [-1,1]^256x256'''
    img = satl.numpy()
    edges = cv2.Canny(
        (denormalize(img, np.nan)[0] * 255).astype(np.uint8),
        thresl, thresh, L2gradient=True
    )
    # expand to stack and normalize to make uniform with other channels
    edges_tensor = tf.expand_dims(tf.convert_to_tensor(edges, dtype=np.float32), axis=-1)
    edges_tensor = normalize(edges_tensor, np.nan)[0]
    ninput = tf.stack(tuple(tf.split(satl, 3, axis=2)) + (edges_tensor,), axis=2) 
    return tf.squeeze(ninput, axis=-1)

def resize(satl, dem, h, w):
    satl = tf.image.resize(
        satl, (h, w), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    dem = tf.image.resize(
        dem, (h, w), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return satl, dem

def random_crop(satl, dem):
    '''Crop input and groundtruth images uniformly'''
    ## tf.split because tf.stack expects same dim inputs
    # NOTE: tried to render split as arbitrary as possible, needs testing
    stacked_image = tf.squeeze( 
        # squeeze because stack returns (H,W,1,Channels) instead of (H,W,Channels)
        tf.stack(tuple(tf.split(satl, satl.shape[-1], axis=-1)) + (dem,), axis=-1)
    )
    cropped_image = tf.image.random_crop(
        stacked_image, size=(IMG_HEIGHT,IMG_WIDTH,stacked_image.shape[-1])
    )

    # expand_dims to yield (h,w,1)
    return cropped_image[...,:-1], tf.expand_dims(cropped_image[...,-1], axis=-1)

def normalize(satl, dem):
    '''Normalize from [0,255] to [-1,1]'''
    return (satl / 127.5) - 1, (dem / 127.5) - 1

def denormalize(satl, dem):
    '''From [-1,1] to [0,1]'''
    return (satl + 1) / 2, (dem + 1) / 2

@tf.function
def random_jitter(satl, dem):
    # resize to 286x286
    satl, dem = resize(satl, dem, 286, 286)
    satl, dem = random_crop(satl, dem)

    # random mirroring
    if tf.random.uniform(()) > 0.5:
        satl = tf.image.flip_left_right(satl)
        dem = tf.image.flip_left_right(dem)

    return satl, dem

def load_db(filename):
    '''Load dataset into two tensors'''
    # load compressed arrays
    data = np.load(filename)
    X, Y = data['arr_0'], data['arr_1']
    
    Xt, Yt = [], []
    for i in range(X.shape[0]):
        if i % 100 == 0: print(i) # process report
        x, y = X[i], Y[i]
        x, y = tf.convert_to_tensor(x, dtype=tf.float32), tf.convert_to_tensor(y, tf.float32)
        x, y = normalize(x, y)
        x, y = random_jitter(x, y)
        x = tf_enhance(x)
        Xt.append(x)
        Yt.append(y)

    return tf.convert_to_tensor(Xt, dtype=tf.float32),\
            tf.convert_to_tensor(Yt, dtype=tf.float32)

# modify filenames accordingly
train_dataset = tf.data.Dataset.from_tensor_slices(
    load_db(f'./Dataset_NoClouds_NoSnow_2000.npz')
)
test_dataset = tf.data.Dataset.from_tensor_slices(
    load_db(f'./TestSet.npz')
)

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 1

## hyperparameters
std_ds = 0.02
std_us = 0.02
drop_per = 0.5
std_gen = 0.02
std_dis = 0.02

def downsample(filters, size, batchnorm=True):
    initializer = tf.random_normal_initializer(0., std_ds)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
        filters, size, strides=2, padding='same', 
        kernel_initializer=initializer, use_bias=False
        )
    )
    if batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, dropout=False):
    initializer = tf.random_normal_initializer(0, std_us)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters, size, strides=2, padding='same',
            kernel_initializer=initializer, use_bias=False
        )
    )
    result.add(tf.keras.layers.BatchNormalization())
    if dropout:
        result.add(tf.keras.layers.Dropout(drop_per))
    result.add(tf.keras.layers.ReLU())

    return result

def Generator(input_channels=INPUT_CHANNELS, output_channels=OUTPUT_CHANNELS):
    '''Generator of ncGAN'''
    down_stack = [
        downsample(64, 4, batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0, std_gen)
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 4, strides=2, padding='same',
        kernel_initializer=initializer, activation='tanh'
    ) # (bs, 256, 256, {output_channels})

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None,None,input_channels])
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator():
    '''Discriminator of cGAN'''
    initializer = tf.random_normal_initializer(0, std_dis)

    inp = tf.keras.layers.Input(shape=[None,None,3], name='input_image')
    tar = tf.keras.layers.Input(shape=[None,None,OUTPUT_CHANNELS], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, 4)

    down1 = downsample(64, 4, batchnorm=False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(
        1, 4, strides=1, kernel_initializer=initializer
    )(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

generator = Generator(INPUT_CHANNELS, OUTPUT_CHANNELS)
discriminator = Discriminator()

LAMBDA = 400

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # l1 loss for crispiness 
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = f'./training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
)

EPOCHS = 150

def generate_images(model, test_input, tar):
    # the training=True is intentional here since
    # we want the batch statistics while running the model
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0][...,:3], tf.squeeze(tar[0]), tf.squeeze(prediction[0])]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    cmaps = [None, plt.cm.binary, plt.cm.binary]

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5, cmap=cmaps[i])
        plt.axis('off')
    plt.show()

@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image[...,:3], target], training=True)
        disc_generated_output = discriminator([input_image[...,:3], gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(
        gen_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

def fit(train_ds, epochs, test_ds, plot=True):
    for epoch in range(epochs):
        start = time.time()

        # Train
        for input_image, target in train_ds.skip(1500):
            train_step(input_image, target)

        if plot:
            try:
                clear_output(wait=True)
            except NameError:
                pass
            # Test on the same image so that the progress of the model can be 
            # easily seen.
            for example_input, example_target in test_ds.take(1):
                generate_images(generator, example_input, example_target)

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'Time taken for epoch {epoch + 1} is {time.time()-start:.1f} sec\n')

fit(train_dataset, EPOCHS, test_dataset)

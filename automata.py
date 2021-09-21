import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout
import models

#Default parameters
CHANNEL_N = 15 # Number of CA state channels
CELL_FIRE_RATE = 0.5
ADD_NOISE = True

class CAModel(tf.keras.Model):

    def __init__(self, model_name, automata_shape, channel_n=CHANNEL_N, extra_chnl=0, fire_rate=CELL_FIRE_RATE,
               add_noise=ADD_NOISE):
        """
        rna_model es un str con el path a la configuración de la RNA
        """

        super().__init__()
        self.automata_shape = automata_shape
        self.channel_n = channel_n
        self.extra_chnl = extra_chnl
        self.fire_rate = fire_rate
        self.add_noise = add_noise
        self.training = False

        self.perceive = tf.keras.Sequential([
            Conv2D(80, 3, activation=tf.nn.relu, padding="SAME"),
        ])
        

        self.dmodel = self.load_model(model_name)

        self(tf.zeros([1, self.automata_shape[0], self.automata_shape[1], channel_n + 4 + self.extra_chnl]))  # dummy calls to build the model

    @tf.function
    def call(self, x, fire_rate=None, manual_noise=None, training=False):
        self.training = training
        img_norm, gray, state = tf.split(x, [3+self.extra_chnl,1, self.channel_n], -1)
        ds = self.dmodel(self.perceive(x))
        if self.add_noise:
            if manual_noise is None:
                residual_noise = tf.random.normal(tf.shape(ds), 0., 0.02)
            else:
                residual_noise = manual_noise
            ds += residual_noise

        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = tf.random.uniform(tf.shape(x[..., 3+self.extra_chnl:4+self.extra_chnl])) <= fire_rate
        living_mask = gray > 0.1
        residual_mask = update_mask & living_mask
        ds *= tf.cast(residual_mask, tf.float32)
        state += ds

        return tf.concat([img_norm, gray, state], -1)

    @tf.function
    def initialize(self, images, normalize=True):
        state = tf.zeros([images.shape[0], self.automata_shape[0], self.automata_shape[1], self.channel_n])
        images = tf.image.resize(images, self.automata_shape, antialias=True)
        images = tf.reshape(images, [-1, self.automata_shape[0], self.automata_shape[1], 4+self.extra_chnl])  
        return tf.cast(tf.concat([images, state], -1), tf.float32)
    @tf.function
    def classify(self, x):
        # The last 10 layers are the classification predictions, one channel
        # per class. Keep in mind there is no "background" class,
        # and that any loss doesn't propagate to "dead" pixels.
        return x[:,:,:,-2:]
    
    def load_model(self, model_name):
        modelo = models.get_hidden(model_name)
        modelo.append(
            Conv2D(self.channel_n, 1, activation=None,
                       kernel_initializer=tf.zeros_initializer)
        )
        return tf.keras.Sequential(modelo)




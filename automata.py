import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import Conv2D, Dropout
from utils import save_plot_loss, save_batch_vis
import models

#Default parameters
CHANNEL_N = 15 # Number of CA state channels
CELL_FIRE_RATE = 0.5
ADD_NOISE = True


def individual_l2_loss(ca, x, y):
    t = y - ca.classify(x)
    return tf.reduce_sum(t**2, [1, 2, 3]) / 2

def batch_l2_loss(ca, x, y):
      return tf.reduce_mean(individual_l2_loss(ca, x, y))

class CAModel(tf.keras.Model):
    lr = 1e-3
    lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [30000, 70000], [lr, lr*0.1, lr*0.01])
    trainer = tf.keras.optimizers.Adam(lr_sched)

    def __init__(self, config):
        """
        rna_model es un str con el path a la configuración de la RNA
        """
        super().__init__()
        self.automata_shape = config['AUTOMATA_SHAPE']
        self.channel_n = config['CHANNEL_N']
        self.extra_chnl = 3 if config['AGREGAR_GIRARD'] else 0
        self.fire_rate = config['CELL_FIRE_RATE']
        self.add_noise = config['ADD_NOISE']
        self.capas_config = config['CAPAS']
        self.config = config

        self.perceive = tf.keras.Sequential([
            Conv2D(80, 3, activation=tf.nn.relu, padding="SAME"),
        ])
        
        raw_model = self.load_yaml_model(self.capas_config) + [Conv2D(self.channel_n, 1, activation=None,
                                                                kernel_initializer=tf.zeros_initializer)]

        self.dmodel = tf.keras.Sequential(raw_model)

        self(tf.zeros([1, self.automata_shape[0], self.automata_shape[1], self.channel_n + 4 + self.extra_chnl]))  # dummy calls to build the model
    
    def load_yaml_model(self, yaml_list):
        _capas = [] 
        for c in yaml_list:
            _capas.append(models.CAPAS_DISP[c[0]](c[1])) #Modificar cuando se tengan más tipos de capas
        return _capas
    
    def guardar_pesos(self, filename):
        self.save_weights(filename)

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
    
    @tf.function
    def train_step(self, x, y):
        iter_n = self.config['N_ITER']
        with tf.GradientTape() as g:
            for i in tf.range(iter_n):
                x = self(x, training=True)
            #print(i, batch_l2_loss(x, y), x.shape, y.shape)
            loss = batch_l2_loss(self, x, y)
        grads = g.gradient(loss, self.weights)
        #print(tf.norm(grads[0]))
        grads = [g/(tf.norm(g)+1e-8) for g in grads]
        self.trainer.apply_gradients(zip(grads, self.weights))
        return x, loss

    def fit(self, x_train, y_train_pic):
        loss_log = []
        for i in range(self.config['EPOCHS']):
            b_idx = np.random.randint(0, x_train.shape[0]-1, size=self.config['BATCH_SIZE'])
            x0 = self.initialize(tf.gather(x_train, b_idx))
            y0 = tf.gather(y_train_pic, b_idx)

            x, loss = self.train_step(x0, y0)

            step_i = len(loss_log)
            loss_log.append(loss.numpy())

            if step_i%10 == 0:
                with open(os.path.join(self.config['SAVE_DIR'], 'loss.txt'), 'w') as f:
                    f.write(', '.join([str(i) for i in loss_log]))
                save_plot_loss(loss, self.config)
                save_batch_vis(self, x0, y0, x, step_i, self.config)

            if step_i%5000 == 0:
                self.guardar_pesos(os.path.join(self.config['SAVE_DIR'], 'model/%07d'%step_i))

        
            print('\r step: %d, log10(loss): %.3f'%(len(loss_log), np.log10(loss)), end='')

        self.guardar_pesos(os.path.join(self.config['SAVE_DIR'], 'model/last'))



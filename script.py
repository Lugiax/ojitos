
import os #5556223216
import io
import PIL.Image, PIL.ImageDraw
import base64
import zipfile
import json
import requests
import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import glob
import re
import pickle
from skimage.transform import resize, rotate
from skimage.util import pad
from skimage.morphology import skeletonize, square, disk, binary_dilation
from skimage.filters.rank import median, sum, mean
from skimage.exposure import adjust_gamma
from skimage.color import gray2rgb, rgb2gray
from skimage.measure import regionprops
from google.protobuf.json_format import MessageToDict
from tensorflow.python.framework import convert_to_constants


import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
os.environ['FFMPEG_BINARY'] = 'ffmpeg'


def np2pil(a):
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1)*255)
    return PIL.Image.fromarray(a)

def imwrite(f, a, fmt=None):
    a = np.asarray(a)
    if isinstance(f, str):
        fmt = f.rsplit('.', 1)[-1].lower()
        if fmt == 'jpg':
            fmt = 'jpeg'
        f = open(f, 'wb')
    np2pil(a).save(f, fmt, quality=95)

def imencode(a, fmt='jpeg'):
    a = np.asarray(a)
    if len(a.shape) == 3 and a.shape[-1] == 4:
        fmt = 'png'
    f = io.BytesIO()
    imwrite(f, a, fmt)
    return f.getvalue()

def im2url(a, fmt='jpeg'):
    encoded = imencode(a, fmt)
    base64_byte_string = base64.b64encode(encoded).decode('ascii')
    return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string

def imshow(a, fmt='jpeg'):
    display(Image(data=imencode(a, fmt)))

def tile2d(a, w=None):
    a = np.asarray(a)
    if w is None:
        w = int(np.ceil(np.sqrt(len(a))))
    th, tw = a.shape[1:3]
    pad = (w-len(a))%w
    a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
    h = len(a)//w
    a = a.reshape([h, w]+list(a.shape[1:]))
    a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))
    return a

def zoom(img, scale=4):
    img = np.repeat(img, scale, 0)
    img = np.repeat(img, scale, 1)
    return img


# Title Data generator
color_lookup = tf.constant([
            [128, 0, 0],  #Para arterias, Rojo
            [0, 128, 128], #Para venas, Azul
            [0, 0, 0], # This is the default for digits/vasos sanguíneos
            [255, 255, 255] # This is the background.
            ])

backgroundWhite = True
def color_labels(x, y_pic, disable_black=False, dtype=tf.uint8):
    # works for shapes of x [b, r, c] and [r, c]
    mask = x[..., -1]
    black_and_white = tf.fill(list(mask.shape) + [2], 0.01)
    is_gray = tf.cast(mask > 0.1, tf.float32)
    is_not_gray = 1. - is_gray
    y_pic = y_pic * tf.expand_dims(is_gray, -1) # forcibly cancels everything outside of it.
  
    # if disable_black, make is_gray super low.
    if disable_black:
        is_gray *= -1e5
        # this ensures that you don't draw white in the digits.
        is_not_gray += is_gray

    bnw_order = [is_gray, is_not_gray] if backgroundWhite else [is_not_gray, is_gray]
    black_and_white *= tf.stack(bnw_order, -1)

    rgb = tf.gather(
      color_lookup,
      tf.argmax(tf.concat([y_pic, black_and_white], -1), -1))
    if dtype == tf.uint8:
        return tf.cast(rgb, tf.uint8)
    else:
        return tf.cast(rgb, dtype) / 255.

#####################################################33
def adjust(img):
    im = 0.8*img[...,1]+0.2*img[...,2]
    def gauss(v, u, sigma = 200):
        return np.exp(-(v-u)**2/(2*sigma**2))
    x = np.arange(im.shape[1])
    y = np.arange(im.shape[0])
    X, Y = np.meshgrid(x,y)
    Z = gauss(X, im.shape[1]//2) + gauss(Y, im.shape[0]//2)
    adjusted = im*Z
    return adjusted / np.max(adjusted)

def get_optical_disk_center(im, t=0.92, scale=0.3): 
    if scale!=1:
        shape_scaled = [int(s*scale) for s in im.shape[:2]]
        im = resize(im, shape_scaled, anti_aliasing=True)
    im = adjust(im)
    im_bin = im > t
    im_bin = binary_dilation(im_bin, disk(10))
    regions = regionprops(im_bin*1)
    return [int(v/scale) for v in regions[0].centroid]# cy, cx

def normalize_girard(batch, sigma=2, selem_size=5):
    def join(r,g,b):
        return np.concatenate((r[..., np.newaxis], g[..., np.newaxis], b[..., np.newaxis]), axis=2)
    new_batch = []
    for j in range(batch.shape[0]):
        rgb = batch[j, ..., :3]
        R = np.uint(rgb[..., 0]*255)
        G = np.uint(rgb[..., 1]*255)
        B = np.uint(rgb[..., 2]*255)
        difs = []
        medians_std = []
        for c in (R,G,B):
            med = median(c, square(selem_size))
            dif = c-med
            difs.append(dif)
            medians_std.append(np.std(dif))

        new_batch.append(join(*[sigma*dif/std+0.5 for dif, std in zip(difs, medians_std)]))

    return np.stack(new_batch) #tf.cast(tf.stack(new_batch), tf.float32)
    
    
#################################################################################################################
root = '/homen1/hector_cam/corridas/ojitos'
folder = os.path.join(root, '../../datos/fondo_ojos')
assert os.path.isdir(folder)

train_path = os.path.join(root, 'disco/d1')
if not os.path.isdir(train_path): 
    os.makedirs(os.path.join(train_path, 'model'))
    os.makedirs(os.path.join(train_path, 'ResultadosClasificaciones/entrenamiento'))
    os.mkdir(os.path.join(train_path, 'ResultadosClasificaciones/prueba'))
    os.mkdir(os.path.join(train_path, 'batches_imgs'))
    os.mkdir(os.path.join(train_path, 'videos'))


#Guardar o cargar la base de datos
accion = 'leer'  #@param ['guardar', 'leer']
datos = 'datos_ventanas_10ppi_1500v_100a_disco' 
automata_shape = (50,50)
archivo = os.path.join(root, f'{datos}.pk')
if accion == 'guardar':
    with open(archivo, 'wb') as f:
        pickle.dump(x_train, f)
        pickle.dump(y_train_pic, f)
        pickle.dump(x_test, f)
        pickle.dump(y_test_pic, f)
    print(f'Archivo guardado en: {archivo}')
else:
    with open(archivo, 'rb') as f:
        x_train = pickle.load(f)
        y_train_pic =pickle.load(f)
        x_test = pickle.load(f)
        y_test_pic = pickle.load(f)
    

new_idx_train = np.random.randint(0, x_train.shape[0]-1, size=x_train.shape[0])
new_idx_test = np.random.randint(0, x_test.shape[0]-1, size=x_test.shape[0])
x_train = tf.image.resize(tf.cast(
         tf.concat((x_train[new_idx_train, ..., :3],
                    normalize_girard(x_train[new_idx_train, ..., :3], selem_size=51),
                    np.ceil(x_train[new_idx_train ,..., -1:])),
                    axis=-1), tf.float32),
                    automata_shape, antialias=True)
y_train_pic = tf.image.resize(tf.cast(y_train_pic[new_idx_train], tf.float32),
                              automata_shape, antialias=True)
x_test = tf.image.resize(tf.cast(
         tf.concat((x_test[new_idx_test, ..., :3],
                    normalize_girard(x_test[new_idx_test, ..., :3], selem_size=51),
                    np.ceil(x_test[new_idx_test ,..., -1:])),
                    axis=-1), tf.float32),
                    automata_shape, antialias=True)
y_test_pic = tf.image.resize(tf.cast(y_test_pic[new_idx_test], tf.float32),
                             automata_shape, antialias=True)

print('Data shapes: ', x_train.shape, y_train_pic.shape, x_test.shape, y_test_pic.shape)


from tensorflow.keras.layers import Conv2D
CHANNEL_N = 10 # Number of CA state channels
BATCH_SIZE = 16
CELL_FIRE_RATE = 0.5
N_ITER = 20

LOSS_TYPE = "l2"

USE_PATTERN_POOL= False
MUTATE_POOL = False
ADD_NOISE = True



class CAModel(tf.keras.Model):

    def __init__(self, channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE,
                   add_noise=ADD_NOISE):
        # CHANNEL_N does *not* include the greyscale channel.
        # but it does include the 10 possible outputs.
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.add_noise = add_noise

        self.perceive = tf.keras.Sequential([
              Conv2D(80, 3, activation=tf.nn.relu, padding="SAME"),
          ])

        self.dmodel = tf.keras.Sequential([
              Conv2D(80, 1, activation=tf.nn.relu),
              Conv2D(self.channel_n, 1, activation=None,
                           kernel_initializer=tf.zeros_initializer),
        ])

        self(tf.zeros([1, 3, 3, channel_n + 7]))  # dummy calls to build the model

    @tf.function
    def call(self, x, fire_rate=None, manual_noise=None):
        img_norm, gray, state = tf.split(x, [6,1, self.channel_n], -1)
        ds = self.dmodel(self.perceive(x))
        if self.add_noise:
            if manual_noise is None:
                residual_noise = tf.random.normal(tf.shape(ds), 0., 0.02)
            else:
                residual_noise = manual_noise
            ds += residual_noise

        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = tf.random.uniform(tf.shape(x[:, :, :, 6:7])) <= fire_rate
        living_mask = gray > 0.1
        residual_mask = update_mask & living_mask
        ds *= tf.cast(residual_mask, tf.float32)
        state += ds
    
        return tf.concat([img_norm, gray, state], -1)

    @tf.function
    def initialize(self, images, normalize=True):
        state = tf.zeros([images.shape[0], automata_shape[0], automata_shape[1], self.channel_n])
        images = tf.image.resize(images, automata_shape, antialias=True)
        images = tf.reshape(images, [-1, automata_shape[0], automata_shape[1], 7])  
        return tf.cast(tf.concat([images, state], -1), tf.float32)
    @tf.function
    def classify(self, x):
        # The last 10 layers are the classification predictions, one channel
        # per class. Keep in mind there is no "background" class,
        # and that any loss doesn't propagate to "dead" pixels.
        return x[:,:,:,-2:]

print(CAModel().perceive.summary(), CAModel().dmodel.summary())

def export_model(ca, base_fn):
    ca.save_weights(base_fn)

    cf = ca.call.get_concrete_function(
      x=tf.TensorSpec([None, None, None, CHANNEL_N+7]),
      fire_rate=tf.constant(0.5),
      manual_noise=tf.TensorSpec([None, None, None, CHANNEL_N]))
    cf = convert_to_constants.convert_variables_to_constants_v2(cf)
    graph_def = cf.graph.as_graph_def()
    graph_json = MessageToDict(graph_def)
    graph_json['versions'] = dict(producer='1.14', minConsumer='1.14')
    model_json = {
      'format': 'graph-model',
      'modelTopology': graph_json,
      'weightsManifest': [],
    }
    with open(base_fn+'.json', 'w') as f:
        json.dump(model_json, f)


# Initialize things for a new training run
ca = CAModel()

def individual_l2_loss(x, y):
    t = y - ca.classify(x)
    return tf.reduce_sum(t**2, [1, 2, 3]) / 2

def batch_l2_loss(x, y):
    return tf.reduce_mean(individual_l2_loss(x, y))

def batch_ce_loss(x, y):
    one_hot = tf.argmax(y, axis=-1)
    # It's ok even if the loss is computed on "dead" cells. Anyway they shouldn't
    # get any gradient propagated through there.
    return tf.compat.v1.losses.sparse_softmax_cross_entropy(one_hot, x)

assert LOSS_TYPE in ["l2", "ce"]
loss_fn = batch_l2_loss if LOSS_TYPE == "l2" else batch_ce_loss

loss_log = []

lr = 1e-3
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      [30000, 70000], [lr, lr*0.1, lr*0.01])
trainer = tf.keras.optimizers.Adam(lr_sched)

@tf.function
def train_step(x, y):
    iter_n = N_ITER
    with tf.GradientTape() as g:
        for i in tf.range(iter_n):
            x = ca(x)
        loss = batch_l2_loss(x, y)
    grads = g.gradient(loss, ca.weights)
    grads = [g/(tf.norm(g)+1e-8) for g in grads]
    trainer.apply_gradients(zip(grads, ca.weights))
    return x, loss


for i in range(1, 10000+1):
    b_idx = np.random.randint(0, x_train.shape[0]-1, size=BATCH_SIZE)
    x0 = ca.initialize(tf.gather(x_train, b_idx))
    y0 = tf.gather(y_train_pic, b_idx)

    x, loss = train_step(x0, y0)


    step_i = len(loss_log)
    loss_log.append(loss.numpy())

    if step_i%5000 == 0:
        export_model(ca, os.path.join(train_path, 'model/%07d'%step_i))


    print('step: %d, log10(loss): %.3f'%(len(loss_log), np.log10(loss)), end='')

export_model(ca, os.path.join(train_path, 'model/last'))


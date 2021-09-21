import numpy as np
import os
import json
import tensorflow as tf
import matplotlib.pylab as pl
import PIL.Image
from google.protobuf.json_format import MessageToDict
from tensorflow.python.framework import convert_to_constants

from skimage.morphology import disk, binary_dilation, square
from skimage.filters.rank import median
from skimage.measure import regionprops
from skimage.transform import resize

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

def cargar_dataset(nombre, carpeta='', agregar_girard=False, selem_size=5):
    with open(os.path.join(carpeta, nombre), 'rb') as f:
        x_train = np.load(f)
        y_train_pic =np.load(f)
        x_test = np.load(f)
        y_test_pic = np.load(f)
    
    new_idx_train = np.random.randint(0, x_train.shape[0]-1, size=x_train.shape[0])
    new_idx_test = np.random.randint(0, x_test.shape[0]-1, size=x_test.shape[0])

    if agregar_girard:
        x_train = tf.cast(
                tf.concat((x_train[new_idx_train, ..., :3],
                            normalize_girard(x_train[new_idx_train, ..., :3], selem_size=selem_size),
                            np.ceil(x_train[new_idx_train ,..., -1:])),
                            axis=-1), tf.float32)
        x_test = tf.cast(
                tf.concat((x_test[new_idx_test, ..., :3],
                            normalize_girard(x_test[new_idx_test, ..., :3], selem_size=selem_size),
                            np.ceil(x_test[new_idx_test ,..., -1:])),
                            axis=-1), tf.float32)
    else:
        x_train = tf.cast(
                tf.concat((x_train[new_idx_train, ..., :3],
                            np.ceil(x_train[new_idx_train ,..., -1:])),
                            axis=-1), tf.float32)
        x_test = tf.cast(
                tf.concat((x_test[new_idx_test, ..., :3],
                            np.ceil(x_test[new_idx_test ,..., -1:])),
                            axis=-1), tf.float32)

    y_train_pic = tf.cast(y_train_pic[new_idx_train], tf.float32)
    y_test_pic = tf.cast(y_test_pic[new_idx_test], tf.float32)

    return x_train, y_train_pic, x_test, y_test_pic



def export_model(ca, base_fn, args):
    ca.save_weights(base_fn)
    with open(os.path.split(base_fn)[0]+'config.txt','w') as f:
        f.write(f"""
CHANNEL_N = {args.n_channels}
CELL_FIRE_RATE = Revisar en automata. Generalmente 0.5
N_ITER = {args.n_iter}
BATCH_SIZE = {args.n_batch}
ADD_NOISE = {args.add_noise}
DATA = {args.db_name}
AGREGAR_GIRARD = {args.agregar_girard}
SELEM_SIZE = {args.selem_size if args.agregar_girard else None}


Arquitectura:
{args.model_name},
Conv2D(self.channel_n, 1, activation=None,
            kernel_initializer=tf.zeros_initializer),
            """)

    cf = ca.call.get_concrete_function(
        x=tf.TensorSpec([None, None, None, args.n_channels+4+args.extra_channels]),
        fire_rate=tf.constant(0.5),
        manual_noise=tf.TensorSpec([None, None, None, args.n_channels]))
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


def classify_and_color(ca, x, y0=None, disable_black=False):
    if y0 is not None:
        return color_labels(
                x[:,:,:,:4], y0, disable_black, dtype=tf.float32)
    else:
        return color_labels(
            x[:,:,:,:4], ca.classify(x), disable_black, dtype=tf.float32)
        

def save_plot_loss(loss_log, args):
    pl.figure(figsize=(6, 3))
    pl.title('Loss history (log10)')
    pl.plot(np.log10(loss_log), '.', alpha=0.4)
    pl.xlabel('Epoch')
    pl.ylabel(f'log10(L)')
    pl.savefig(os.path.join(args.save_dir, 'figures', 'loss.png'))


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

def save_batch_vis(ca, x0, y0, x, step_i, args):
    vis_1 = np.hstack(x0[..., :3])
    vis0 = np.hstack(classify_and_color(ca, x0, y0).numpy())
    vis1 = np.hstack(classify_and_color(ca, x).numpy())
    vis = np.vstack([vis_1, np.hstack(x0[..., 3:6]), vis0, vis1]) if args.agregar_girard else np.vstack([vis_1, vis0, vis1])
    imwrite(os.path.join(args.save_dir, 'figures', 'batches_%04d.jpg'%step_i), vis)
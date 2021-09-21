import argparse
import tensorflow as tf
import numpy as np
import os

from skimage.morphology import selem
from automata import CAModel 
from utils import cargar_dataset, export_model, save_plot_loss, save_batch_vis

parser = argparse.ArgumentParser()
parser.add_argument('db_name', type=str,
                    help='Nombre de la base de datos')
parser.add_argument('model_name', type=str,
                    help='Nombre del modelo de la RNA')
parser.add_argument('--db_dir', type=str,
                    help='Directorio de la base de datos',
                    default='dbs')
parser.add_argument('--n_channels', type=int,
                    help='Número de canales',
                    default=10)
parser.add_argument('--n_iter', type=int,
                    help='Número de iteraciones del autómata',
                    default=20)
parser.add_argument('--n_batch', type=int,
                    help='Número de elementos por lote',
                    default=4)
parser.add_argument('--selem_size', type=int,
                    help='Tamaño del elemento morfológico para normalización de Girard',
                    default=5)
parser.add_argument('--n_epochs', type=int,
                    help='Número de épocas de entrenamiento',
                    default=10000)
parser.add_argument('--save_dir', type=str,
                    help='Directorio donde se guardarán todos los archivos generados',
                    default='resultados')
parser.add_argument('--add_noise', action='store_true',
                    help='Bandera para agregar ruido al entrenamiento') 
parser.add_argument('--agregar_girard', action='store_true',
                    help='Bandera para agregar normalización de Girard')                      

args = parser.parse_args()

args.extra_channels = 3 if args.agregar_girard else 0

if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)
    os.mkdir(os.path.join(args.save_dir, 'figures'))
    os.mkdir(os.path.join(args.save_dir, 'model'))

x_train, y_train_pic, x_test, y_test_pic = cargar_dataset(nombre=args.db_name,
                                                          carpeta=args.db_dir,
                                                          agregar_girard=args.agregar_girard,
                                                          selem_size=args.selem_size)


ca = CAModel(model_name=args.model_name,
             automata_shape=tuple(x_train.shape[1:3]),
             channel_n=args.n_channels,
             extra_chnl=args.extra_channels,
             add_noise=args.add_noise)

def individual_l2_loss(x, y):
    t = y - ca.classify(x)
    return tf.reduce_sum(t**2, [1, 2, 3]) / 2

def batch_l2_loss(x, y):
    return tf.reduce_mean(individual_l2_loss(x, y))

loss_log = []
lr = 1e-3
lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      [30000, 70000], [lr, lr*0.1, lr*0.01])
trainer = tf.keras.optimizers.Adam(lr_sched)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as g:
        for i in tf.range(args.n_iter):
            x = ca(x, training=True)
        loss = batch_l2_loss(x, y)
    grads = g.gradient(loss, ca.weights)
    grads = [g/(tf.norm(g)+1e-8) for g in grads]
    trainer.apply_gradients(zip(grads, ca.weights))
    return x, loss

for i in range(args.n_epochs):
    b_idx = np.random.randint(0, x_train.shape[0]-1, size=args.n_batch)
    x0 = ca.initialize(tf.gather(x_train, b_idx))
    y0 = tf.gather(y_train_pic, b_idx)

    x, loss = train_step(x0, y0)

    step_i = len(loss_log)
    loss_log.append(loss.numpy())

    if step_i%100 == 0:
        with open(os.path.join(args.save_dir, 'loss.txt'), 'w') as f:
            f.write(', '.join([str(i) for i in loss_log]))
        save_plot_loss(loss, args)
        save_batch_vis(ca, x0, y0, x, step_i, args)

    if step_i%5000 == 0:
        export_model(ca,
                    os.path.join(args.save_dir, 'model/%07d'%step_i),
                    args)

  
    print('\r step: %d, log10(loss): %.3f'%(len(loss_log), np.log10(loss)), end='')

export_model(ca,
             os.path.join(args.save_dir, 'model/last'),
             args)
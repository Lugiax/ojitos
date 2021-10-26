import argparse
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
import shutil

from scipy.io import savemat
from time import time
from automata import CAModel 
from utils import cargar_dataset, export_model, save_plot_loss,\
                  save_batch_vis, imwrite, color_labels, calcular_metricas

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str,
                    help='Archivo de configuracíón .yaml')
parser.add_argument('save_dir', type=str,
                    help='Directorio donde se guardarán todos los archivos generados',
                    default='resultados')
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
parser.add_argument('--n_eval_imgs', type=int,
                    help='Número de imágenes de evaluación de los conjuntos de entrenamiento y prueba',
                    default=None)
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
    os.makedirs(os.path.join(args.save_dir, 'ResultadosEval/entrenamiento'))
    os.makedirs(os.path.join(args.save_dir, 'ResultadosEval/prueba'))


#Lectura del archivo config
with open(args.config, 'r') as f:
    config = yaml.load(f)
#Se crea una copia del archivo config en los resultados
shutil.copy(args.config, args.save_dir)

x_train, y_train_pic, x_test, y_test_pic = cargar_dataset(fname=config['DB_FNAME'],
                                                          agregar_girard=config['AGREGAR_GIRARD'],
                                                          selem_size=config['SELEM_SIZE'])

config['SAVE_DIR'] = args.save_dir
config['AUTOMATA_SHAPE'] = tuple(x_train.shape[1:3])

"""
fig, axs = plt.subplots(5, 3)
color_labels_list = color_labels(x_train[:5], y_train_pic[:5])
for k in range (5):
    axs[k, 0].imshow(x_train[k, ..., :3])
    axs[k, 1].imshow(x_train[k, ..., -1], cmap='gray')
    axs[k, 2].imshow(color_labels_list[k])
plt.show()
"""

ca = CAModel(config)
ca.fit(x_train, y_train_pic)

print(f'Modelo entrenado y guardado en {os.path.join(config["SAVE_DIR"], "model/last")}'
       '\nSe procede a realizar la evaluación')

## Generación de las métricas

#Métricas del conjunto de prueba:
#(Se hace un recorrido para no saturar la memoria)
metricas = None
for k in range(x_train.shape[0]):
    y_pred = ca.predict(x_train[k], binary=True)
    if metricas is None:
        metricas = calcular_metricas(y_)


#Evaluación del modelo
n_imgs_test = x_train.shape[0] if args.n_eval_imgs is None else args.n_eval_imgs
for save_path, x, y, tipo in [[os.path.join(config['SAVE_DIR'], 'ResultadosEval/'),
                         x_train[:n_imgs_test],
                         y_train_pic[:n_imgs_test],
                         'entrenamiento'],
                        [os.path.join(config['SAVE_DIR'], 'ResultadosEval/'),
                         x_test[:n_imgs_test],
                         y_test_pic[:n_imgs_test],
                         'prueba']]:
    metricas = [dict(),dict()]
    intervalos = 10
    for i in range(0, x.shape[0], intervalos):
        sub_x, sub_y = x[i : i+intervalos], y[i : i+intervalos]
        y_pred = ca.predict(sub_x)
        _metricas = calcular_metricas(sub_y, y_pred)
        for tipo_av in [0,1]:
            for res in _metricas:
                metricas[tipo_av][res] = metricas[tipo_av].get(res, []) + _metricas[res][tipo_av]

    for k, tipo_av in enumerate(['arterias', 'venas']):
        with open(os.path.join(save_path,f'{tipo}_{tipo_av}.yaml'), 'w') as f:
            f.write('metrica,promedio,min,max,std\n')
            for m in metricas[k]:
                data = metricas[k][m]
                f.write(','.join([m,str(np.mean(data)),
                                    str(min(data)),
                                    str(max(data)),
                                    str(np.std(data))]) + '\n')
          
    """
    print(f'\tEvaluando conjunto de {tipo}')
    t1 = time()
    x0 = ca.initialize(x)
    for _ in range(20):
        x0 = ca(x0)
    t2= time()
    print(f'\t  Tiempo procesado de {t2-t1} segundos')


    true_labels = color_labels(x, y)
    pred_labels = color_labels(x, x0[..., -2:])

    matrices = {}
    for j in range(x.shape[0]):
        art_d, ven_d = y[j, ..., 0], y[j, ..., 1]
        art_p, ven_p = x0[j, ..., 0], x0[j, ..., 1]
        imwrite(os.path.join(save_path, f'gt_art_{j}.png'), y[j, ..., 0])
        imwrite(os.path.join(save_path, f'gt_ven_{j}.png'), y[j, ..., 1])
        imwrite(os.path.join(save_path, f'gtcolor_{j}.png'), true_labels[j])
        imwrite(os.path.join(save_path, f'pd_art_{j}.png'), x0[j, ..., -2])
        imwrite(os.path.join(save_path, f'pd_ven_{j}.png'), x0[j, ..., -1])
        imwrite(os.path.join(save_path, f'pdcolor_{j}.png'), pred_labels[j])

        savemat(os.path.join(save_path, f'matrices_{j}.m'),
                {'gt_art' : y[j, ..., 0],
                'gt_ven' : y[j, ..., 1],
                'gtcolor': true_labels[j],
                'pd_art' : x0[j, ..., -2],
                'pd_ven' : x0[j, ..., -1],
                'pdcolor': pred_labels[j]}
                )
    """

print(f'Resultados de la evaluación guardados en {os.path.join(config["SAVE_DIR"], "ResultadosEval/")}')
import argparse
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
import shutil

from scipy.io import savemat
from PIL import Image
from time import time
from automata import CAModel, Automata2
from utils import cargar_dataset, abrir_img, calcular_metricas, classify_and_color
from glob import glob

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str,
                    help='Archivo de configuracíón .yaml')
parser.add_argument('operacion', type=str,
                    help='Tipo de operación a realizar: inferencia o entrenamiento')
parser.add_argument('--save_dir', type=str,
                    help='Directorio donde se guardarán todos los archivos generados',
                    default='resultados')
parser.add_argument('--model_path', type=str,
                    help='Ruta al archivo del modelo, necesario para inferencia')
parser.add_argument('--img_path', type=str,
                    help='El acceso a la ruta de la o las imágenes a procesar.')
#parser.add_argument('--db_dir', type=str,
#                    help='Directorio de la base de datos',
#                    default='dbs')
#parser.add_argument('--n_channels', type=int,
#                    help='Número de canales',
#                    default=10)
#parser.add_argument('--n_iter', type=int,
#                    help='Número de iteraciones del autómata',
#                    default=20)
#parser.add_argument('--n_batch', type=int,
#                    help='Número de elementos por lote',
#                    default=4)
#parser.add_argument('--selem_size', type=int,
#                    help='Tamaño del elemento morfológico para normalización de Girard',
#                    default=5)
#parser.add_argument('--n_epochs', type=int,
#                    help='Número de épocas de entrenamiento',
#                    default=10000)
parser.add_argument('--n_eval_imgs', type=int,
                    help='Número de imágenes de evaluación de los conjuntos de entrenamiento y prueba',
                    default=None)
#parser.add_argument('--add_noise', action='store_true',
#                    help='Bandera para agregar ruido al entrenamiento') 
#parser.add_argument('--agregar_girard', action='store_true',
#                    help='Bandera para agregar normalización de Girard')                      

args = parser.parse_args()


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

print('Inicio del proceso')
print(f'Configurado para {args.operacion}')
##Se cargan las imágenes
if args.operacion=='entrenamiento':
    print(f'Dataset {config["DB_FNAME"]}')
    #pdb.set_trace()
    x_train, y_train_pic, x_test, y_test_pic = cargar_dataset(fname=config['DB_FNAME'],
                                                            agregar_girard=config['AGREGAR_GIRARD'],
                                                            selem_size=config['SELEM_SIZE'])
                                       
    n_canales = x_train.shape[-1]
    config['AUTOMATA_SHAPE'] = tuple(x_train.shape[1:3])

elif args.operacion=='inferencia':
    #Se lee el archivo a procesar.
    formatos_validos = ('jpg', 'jpeg', 'tif', 'tiff', 'png')
    lista_imgs_paths = glob(os.path.join(args.img_path, '*.*')) if os.path.isdir(args.img_path)\
                                                                else [args.img_path]
    infer_imgs = [abrir_img(impath) for impath in lista_imgs_paths\
                                    if impath.split('.')[-1] in formatos_validos]
    n_canales = infer_imgs[0].shape[-1]
                                        
##Se revisa si los datos tienen 3 canales (sin máscara), y que el parámetro en la configuración
##esté bien colocado... Aunque se podría quitar del archivo de configuración y definir aquí

assert config['SIN_MASCARA'] and n_canales==3, 'Parece que tu configuración no es correcta.'
config['SAVE_DIR'] = args.save_dir

#Se arma el modelo
print('Armando el modelo...')
ca = [CAModel, Automata2][int(config['SIN_MASCARA'])](config)


if args.operacion == 'entrenamiento':
    ca.fit(x_train, y_train_pic)

    print(f'Modelo entrenado y guardado en {os.path.join(config["SAVE_DIR"], "model/last")}'
        '\nSe procede a realizar la evaluación')

    ## Generación de las métricas

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
            sub_x, sub_y = x[i : i+intervalos], y[i : i+intervalos]>0.1
            y_pred = ca.predict(sub_x, binary=True)
            _metricas = calcular_metricas(sub_y, y_pred)
            for tipo_av in [0,1]:
                for res in _metricas:
                    metricas[tipo_av][res] = metricas[tipo_av].get(res, []) + _metricas[res][tipo_av]

        for k, tipo_av in enumerate(['arterias', 'venas']):
            with open(os.path.join(save_path,f'{tipo}_{tipo_av}.csv'), 'w') as f:
                f.write('metrica,promedio,min,max,std\n')
                for m in metricas[k]:
                    data = metricas[k][m]
                    f.write(','.join([m,str(np.mean(data)),
                                        str(min(data)),
                                        str(max(data)),
                                        str(np.std(data))]) + '\n')


    print(f'Resultados de la evaluación guardados en {os.path.join(config["SAVE_DIR"], "ResultadosEval/")}')

elif args.operacion == 'inferencia':
    print('Realizando la inferencia')
    x = tf.constant(infer_imgs)
    y = ca.predict(x)
    print('Coloreando')
    res = classify_and_color(ca, x, y)
    print(res.shape)
    print('Guardando los resultados')
    for k in range(res.shape[0]):
        im_name = os.path.basename(lista_imgs_paths[k]).split('.')
        im_name = ''.join([im_name[0], '_proc.png'])
        im_path = os.path.join(args.save_dir, im_name)
        arr_img = (res[k].numpy()*255).astype(np.uint8)
        _im = Image.fromarray(arr_img)

        #pdb.set_trace()
        _im.save(im_path)
        
import os
import argparse
import PIL.Image
import re
import numpy as np
import matplotlib.pyplot as plt
import pdb
import time

from glob import glob
from skimage.transform import resize, rotate
from skimage.morphology import skeletonize, disk, binary_dilation
from skimage.measure import regionprops
from skimage.io import imread

from utils import abrir_img


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str,
                    help='Directorio donde se encuentran las imágenes')
parser.add_argument('--save_dir', type=str,
                    help='Directorio donde se guardarán todos los archivos generados',
                    default='dbs')
#parser.add_argument('--nombre', type=str,
#                    help='Nombre del archivo',
#                    default='data.npy')
parser.add_argument('--n_aumentados', type=int,
                    help='Cantidad de generados por imagen',
                    default=1)
parser.add_argument('--tipo', type=str,
                    help='Basado en: disco óptico o ventanas. Seleccionar de entre: [disco, ventanas]',
                    default='disco')
parser.add_argument('--angulo_max', type=int,
                    help='Máximo ángulo de rotación de los nuevos datos, debe ser mayor a 1deg',
                    default=1)
parser.add_argument('--desp_max', type=int,
                    help='Máximo desplazamiento del centro de la imagen',
                    default=1)
parser.add_argument('--tamano_vent', type=int,
                    help='Tamaño en px de las ventanas. Recomendado 800 para disco y 300 para ventanas',
                    default=800)
parser.add_argument('--padding', type=int,
                    help='Padding para las zonas negras en la rotación',
                    default=None)
parser.add_argument('--automata_shape', type=str,
                    help='Dimensiones del autómata en la forma de cadena (H,W)',
                    default='(50,50)')
parser.add_argument('--mostrar', action='store_true',
                    help='Bandera para mostrar los datos generados')
parser.add_argument('--sin_mascara', action='store_true',
                    help='Genera la BD sin el canal de la máscara binaria')                    

args = parser.parse_args()



source = args.data_dir
save_path  = args.save_dir
assert os.path.isdir(source), 'No existe el directorio de datos'
if not os.path.isdir(save_path):
    os.makedirs(save_path)


# Parámetros:
N_aumentos_x_img = args.n_aumentados
max_angle = args.angulo_max
tipo = args.tipo
assert tipo in ['disco', 'ventanas']
ventana = args.tamano_vent
padding = args.padding #Si es None, se considera una padding igual a ventana//3
               #El padding evita partes negras en las imágenes de los datos
desplazamiento = args.desp_max
automata_shape = eval(args.automata_shape)
assert len(automata_shape)==2

## ------------------ FUNCS
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

def leer_centro(path):
    with open(path, 'r') as f:
        lineas = f.readlines()
        coord = [int(c.strip()) for c in lineas[1].split(',')]
        return coord[::-1]


# -------------------------LOOP
x = []
y = []
padding = ventana if padding is None else padding
sub_pad = ventana//3+padding

lista_rgb = []
lista_labels = []
lista_centros = []
db_source_name = None

##Para el procesamiento de datos se almacenan los nombres de los archivos
##de las imágenes RGB y de las etiquetas.

## Procesamiento para la base de datos del IIMAS
if 'iimas' in source.lower():
    lista_archivos = glob(source + '/*')
    nombres_unicos = {re.findall(r'(\d+_\d+)', a)[0] for a in lista_archivos}
    dict_archivos = {u: [a for a in lista_archivos if os.path.basename(a).startswith(u)] for u in nombres_unicos}
    for n, k in enumerate(dict_archivos):#enumerate(list(dict_archivos.keys())[:10]):#
        labels = []
        for i in dict_archivos[k]:
            if 'artery' in i: #Máscara de arterias
                labels.insert(0, i)
            elif 'vein' in i: #Máscara de venas
                labels.append(i)
            elif 'DO' in i:
                lista_centros.append(i)
            else: ## La imagen RGB
                lista_rgb.append(i)
        lista_labels.append(labels)
    db_source_name = 'iimas'
        
        
##Procesamiento de datos de HRF-AV
##Dentro de esta carpeta tenemos tres más, sólo se usan dos de ellas.
elif 'hrf' in source.lower():
    lista_rgb = sorted(glob(source + '/Images/*'))
    lista_labels = sorted(glob(source + '/AVReference/*'))
    db_source_name = 'hrf'

elif 'rite' in source.lower():
    lista_rgb = sorted(glob(source + '/images/*'))
    lista_labels = sorted(glob(source + '/av/*'))
    db_source_name = 'rite'
    if ventana>300:
        print('El tamaño de las imágenes de RITE-DRIAVE es de 500x500px. '
              'Considera otro tamaño de ventana')

else:
    raise ValueError('Carpeta incorrecta')

#Proceso de carga de imágenes y generación del dataset
for rgb_path, labels_path, centro_path in zip(lista_rgb, lista_labels, lista_centros):
    print(f'Trabajando con la imagen {os.path.basename(rgb_path).split(".")[0]}') 
    ##Image lecture

    rgb_img = abrir_img(rgb_path)/255.

    if 'iimas' in source.lower():
        labels = [
            abrir_img(labels_path[0])*1.,
            abrir_img(labels_path[1])*1.
        ]
    elif 'hrf' in source.lower() or 'rite' in source.lower():
        label_img = abrir_img(labels_path)/255.
        #print(f'Label inter {np.min(label_img)} - {np.max(label_img)}')
        labels = [
                label_img[..., 0],
                label_img[..., 2]
                #np.clip(label_img[..., 0] + label_img[..., 1], 0, 1), #R + G
                #np.clip(label_img[..., 2] + label_img[..., 1], 0, 1) #B + G
                ]

    ##Image padding
    rgb_img = np.pad( rgb_img, ((padding,padding),(padding,padding), (0,0)) )
    labels = [np.pad(l, padding) for l in labels]
    
    if tipo=='disco':
        print('\tTrabajando con el disco óptico... ', end='')
        #cy, cx = get_optical_disk_center(rgb_img, scale=0.3)
        cy, cx = leer_centro(centro_path)
        print(f'leído en {cy} y - {cx} x')
        desps = np.random.randint(-desplazamiento, desplazamiento, size=(N_aumentos_x_img, 2))
        coords = [(cy+dy, cx+dx) for dy,dx in desps]
    else:
        #Se buscan las coordenadas donde hay vasos sanguíneos
        print(f'\t- Buscando las coordenadas válidas para cada vaso de búsqueda...')
        mask = np.uint((labels[0] + labels[1])/2. > 0)
        esqueleto = skeletonize(mask)
        coords = list(zip(*np.where(esqueleto)))
        np.random.shuffle(coords)
        print(f'\t\t {len(coords)} coordenadas encontradas')


    print('\t- Generando datos...', end='')
    counter = 0
    selected = 0
    while counter <N_aumentos_x_img and selected < len(coords) :
        if N_aumentos_x_img>=10 and counter % int(N_aumentos_x_img*0.1)==0:
            print(f'\r\t- Generando datos... Encontrados: {counter}', end='')

        cy, cx = coords[selected]
        selected += 1
        
        angle = float(np.random.randint(-max_angle, max_angle))
        y_min = max(0, cy-sub_pad)
        x_min = max(0, cx-sub_pad)

        rot_img = rotate(rgb_img[y_min:cy+sub_pad,x_min:cx+sub_pad], angle)
        new_img = resize(rot_img[padding:-padding, padding:-padding],
                            automata_shape,
                            anti_aliasing=True)

        rot_labels = [rotate(l2, angle) for l2 in
                        [l1[y_min:cy+sub_pad, x_min:cx+sub_pad] for l1 in labels]
                        ]
                                
        new_labels = [resize(l[padding:-padding, padding:-padding],
                                    automata_shape,
                                    anti_aliasing=True)\
                        for l in rot_labels]
        
        mascara_completa = new_labels[0] + new_labels[1] > 0.1

        if args.mostrar:
            fig, axs = plt.subplots(1,2)
            axs[0].imshow(new_img)
            axs[1].imshow(mascara_completa)
            plt.draw()
            plt.plot()
            time.sleep(5)

        if args.sin_mascara:
            x.append(new_img)
        else:
            x.append(np.concatenate( [new_img, mascara_completa[..., np.newaxis]], axis=2))
        y.append(np.stack(new_labels, axis=2))

        counter+=1
        print(new_img.shape)
    print(f'\r\t- Generando datos... {counter} datos generados... Continuando...')

print(f'Terminado :D ... generados {len(x)} datos')        
print('Partiendo los datos')        
n_test = int(0.2*len(x))
y_train_pic = np.stack(y[:-n_test], axis=0)
x_train = np.stack(x[:-n_test])
y_test_pic = np.stack(y[-n_test:], axis=0)
x_test = np.stack(x[-n_test:])
print('Datos partidos :D')

del x,y

print('Dimensiones: xtrain: {}, ytrain: {}, xtest: {}, ytest: {}'.format(x_train.shape,
                                                                        y_train_pic.shape, 
                                                                        x_test.shape, 
                                                                        y_test_pic.shape))


##Se le da forma al nombre del archivo
nombre = f'{db_source_name}_{tipo}_{N_aumentos_x_img}aug_{ventana}ven_'\
         f'{automata_shape[0]}aut_{max_angle}deg_{desplazamiento}des_'\
         f'{"no-" if args.sin_mascara else ""}mask.npy'
save_fn = os.path.join(save_path, nombre)

with open(save_fn, 'wb') as f:
    np.save(f, x_train)
    np.save(f, y_train_pic)
    np.save(f, x_test)
    np.save(f, y_test_pic)
    print(f'Archivo guardado en: {save_fn}')
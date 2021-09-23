import os
import argparse
import PIL.Image
import re
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from skimage.transform import resize, rotate
from skimage.morphology import skeletonize, disk, binary_dilation
from skimage.measure import regionprops


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
                    default=500)
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
                    help='Dimensiones del autómata en la forma (H,W)',
                    default='(50,50)')
parser.add_argument('--mostrar', action='store_true',
                    help='Bandera para mostrar los datos generados')                    

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
mask_thresh = 0.15
ventana = args.tamano_vent
padding = args.padding #Si es None, se considera una padding igual a ventana//3
               #El padding evita partes negras en las imágenes de los datos
desplazamiento = args.desp_max
automata_shape = eval(args.automata_shape)
assert len(automata_shape)==2

##Se le da forma al nombre
nombre = f'datos_{tipo}_{N_aumentos_x_img}aug_{ventana}ven_{automata_shape[0]}aut_{max_angle}deg_{desplazamiento}des.npy'
save_fn = os.path.join(save_path, nombre)

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


def abrir_img(path, shape=None):
    img = PIL.Image.open(path)
    if shape is not None:
        img = img.resize(shape)
    return np.array(img)


# -------------------------LOOP
if args.mostrar: 
    fig, axs = plt.subplots(1,3)
    #plt.ion()
    #plt.show()
x = []
y = []
padding = ventana if padding is None else padding
sub_pad = ventana//2+padding

lista_rgb = []
lista_labels = []

## Procesamiento para la base de datos del IIMAS
if 'iimas' in source.lower():
    lista_archivos = glob(source + '/*')
    nombres_unicos = {re.findall(r'(\d+_\d+)', a)[0] for a in lista_archivos}
    dict_archivos = {u: [a for a in lista_archivos if os.path.basename(a).startswith(u)] for u in nombres_unicos}
    for n, k in enumerate(dict_archivos):#enumerate(list(dict_archivos.keys())[:10]):#
        print(f'Abriendo imagen {n+1} ({k})')
        labels = []
        for i in dict_archivos[k]:
            if 'artery' in i: #Máscara de arterias
                b = abrir_img(i)*1.#tf.cast(abrir_img(i, img_shape)*1, tf.float32)
                b= np.pad(b, padding, mode='constant', constant_values=0)
                labels.insert(0, b)
            elif 'vein' in i: #Máscara de venas
                b = abrir_img(i)*1.#tf.cast(abrir_img(i, img_shape)*1, tf.float32)
                b= np.pad(b, padding, mode='constant', constant_values=0)
                labels.append(b)
            else:
                img = abrir_img(i)/255.
                img = np.pad(img,((padding,padding),(padding,padding), (0,0)), mode='constant', constant_values=0)
        lista_labels.append(labels)
        lista_rgb.append(img)
        
##Procesamiento de datos de HRF-AV
##Dentro de esta carpeta tenemos tres más, sólo se usan dos de ellas.
elif 'hrf' in source.lower():
    lista_rgb = sorted(glob(source + '/Images/*'))
    lista_cls = sorted(glob(source + '/AVReference/*'))
    print('Abriendo imgs')
    rgb_imgs = [
                np.pad(abrir_img(i)/255.,
                        ((padding,padding),(padding,padding), (0,0)))
                for i in lista_rgb
                ]
    cls_imgs = [np.pad(abrir_img(i)*1., padding) for i in lista_cls]
    print('Abriendo clasifs')
    lista_labels = [
                    (np.clip(cls_img[..., 0] + cls_img[..., 1], 0, 1), #R + G
                    np.clip(cls_img[..., 2] + cls_img[..., 1], 0, 1)) #B + G
                    for cls_img in cls_imgs
                ]
else:
    raise ValueError('Carpeta incorrecta')

for rgb_img, labels in zip(lista_rgb, lista_labels): 

    if tipo=='disco':
        print('\tTrabajando con el disco óptico... ', end='')
        cy, cx = get_optical_disk_center(img, scale=0.3)
        print(f'encontrado en {cy} y - {cx} x')
        desps = np.random.randint(-desplazamiento, desplazamiento, size=(N_aumentos_x_img, 2))
        coords = [(cy+dy, cx+dx) for dy,dx in desps]
        #img[cy-10:cy+11, cx-10:cx+11] = np.ones((21,21,3))
    else:
        #Se buscan las coordenadas donde hay vasos sanguíneos
        print(f'\t- Buscando las coordenadas válidas para cada vaso de búsqueda...')
        mask = np.uint((labels[0] + labels[1])/2. > 0)
        esqueleto = skeletonize(mask)
        coords = list(zip(*np.where(esqueleto)))
        np.random.shuffle(coords)
        print(f'\t\t {len(coords)} coordenadas encontradas')

    #plt.imshow(img)
    #plt.plot([cx], [cy], 'g*')
    #plt.show()
    print('\t- Generando datos... Encontrados:', end='')
    counter = 0
    selected = 0
    while counter <N_aumentos_x_img and selected < len(coords) :
        if counter% int(N_aumentos_x_img*0.1)==0:
            print(f'{counter}', end='-')
        #try:
        cy, cx = coords[selected]
        selected += 1
        
        angle = float(np.random.randint(-max_angle, max_angle))
        rot_labels = [rotate(l2, angle) for l2 in #
                        [l1[cy-sub_pad:cy+sub_pad,cx-sub_pad:cx+sub_pad] for l1 in labels]
                        ]   
        new_labels = [resize(l[padding:-padding, padding:-padding],
                                    automata_shape,
                                    anti_aliasing=True)\
                        for l in rot_labels]
        
        mascara_completa = (new_labels[0] + new_labels[1])/2. >= mask_thresh
        #print(np.mean(mascara_completa))
        #_, ax = plt.subplots(1,2)
        #ax[0].imshow((new_labels[0] + new_labels[1])/2)
        #ax[1].imshow(mascara_completa)
        #plt.show()
        #Si no hay al menos el 10% de vasos en la image, salta al siguiente
        #if np.mean(mascara_completa) < 0.1:
        #    continue
        
        rot_img = rotate(img[cy-sub_pad:cy+sub_pad,cx-sub_pad:cx+sub_pad], angle)
        
        new_img = resize(rot_img[padding:-padding, padding:-padding],
                            automata_shape,
                            anti_aliasing=True)


        if args.mostrar:
            axs[0].imshow(new_img)
            axs[1].imshow(mascara_completa)
            axs[2].imshow(new_img*np.repeat(mascara_completa[..., np.newaxis], 3, axis=2))
            plt.draw()
            plt.pause(0.5)

        x.append(np.concatenate( [new_img, mascara_completa[..., np.newaxis]], axis=2))
        y.append(np.stack(new_labels, axis=2))

        counter+=1
        #except:
        #    pass
    print(f'... {counter} datos generados... Continuando...')

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

with open(save_fn, 'wb') as f:
    np.save(f, x_train)
    np.save(f, y_train_pic)
    np.save(f, x_test)
    np.save(f, y_test_pic)
    print(f'Archivo guardado en: {save_fn}')
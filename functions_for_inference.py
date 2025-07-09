import json
import  numpy as np
np.random.seed(2312)
from model import PGAN
import tensorflow as tf
from typing import Callable
from argparse import ArgumentParser

config_filepath = 'config.json'
with open(config_filepath, "r") as f:
    config_file = json.load(f)

train_config = config_file.get('train_config')
model_config = config_file.get('model_config')

def parser(
    prog_name: str, dscr: str, get_args: Callable[[ArgumentParser], ArgumentParser]
) -> Callable[[Callable], Callable]:
    def decorator(function):
        def new_function(*args, **kwargs):
            prs = ArgumentParser(
                prog=prog_name,
                description=dscr,
            )

            prs = get_args(prs)
            args = prs.parse_args()
            function(args)

        return new_function

    return decorator

### ----------------------------------------- ###
#|                                             |#
#|              UTILS FUNCTIONS                |#
#|                                             |#
### ----------------------------------------- ###

def dynamic_range_opt(array, epsilon=train_config['epsilon'], mult_factor=train_config['mult_factor']):
    array = (array + epsilon)/epsilon
    a = np.log10(array)
    b = np.log10(1/epsilon)
    return a/b * mult_factor

def inv_dynamic_range(synth_img, eps=train_config['epsilon'], mult_factor=train_config['mult_factor']):
    image = np.asarray(synth_img)/mult_factor
    a = - eps**(1 - image)
    b = eps**image - 1
    return a*b

def new_rescaling(
        gan_map:np.ndarray,
        mass:float
        ) -> np.ndarray:
    Dian_sim_fit = lambda m: -13.2 + 0.79*(m)
    ref_value = 10**Dian_sim_fit(mass)
    initial_value = np.sum(gan_map)

    return gan_map * (ref_value / initial_value)

def generate_image(
        mass:float, model:PGAN, latent_vector:np.ndarray
        ) -> np.ndarray:
    
    mass_tensor = tf.convert_to_tensor(np.expand_dims(mass, 0), dtype=tf.float32)
    random_latent_vectors = tf.convert_to_tensor(np.expand_dims(latent_vector, 0), dtype=tf.float32)
    image = model.generator([random_latent_vectors, mass_tensor])
    
    return np.squeeze(inv_dynamic_range(image))

def Fourier_transform(
        image:np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
    '''This function simply applies np.fft.rfft2 and returns real and imaginary part.'''
    image_fft = np.fft.rfft2(image)

    return np.real(image_fft), np.imag(image_fft)

### ----------------------------------------- ###
#|                                             |#
#|          GAN MODEL INITIALIZATION           |#
#|                                             |#
### ----------------------------------------- ###

def build_gan(
        ckpt_path:str
        ) -> PGAN:
    
    end_size = train_config['end_size']
    xgan = PGAN(pgan_config=model_config, version=config_file['version'])
    for n_depth in range(1, int(np.log2(end_size/2))):
        xgan.n_depth = n_depth

        xgan.fade_in_generator()
        xgan.fade_in_discriminator()

        xgan.stabilize_generator()
        xgan.stabilize_discriminator()

    xgan.build(input_shape=(end_size, end_size))
    xgan.load_weights(ckpt_path)
    
    return xgan

### ----------------------------------------- ###
#|                                             |#
#|              FOURIER SPACE MASK             |#
#|                                             |#
### ----------------------------------------- ###

def build_mask_structure(matrix_size):

    mask = np.ones((matrix_size//2, matrix_size//2))
    u, v = np.arange(mask.shape[0]), np.arange(mask.shape[1])

    uu, vv = np.meshgrid(u, v, indexing='ij')
    mask[vv > matrix_size - 2 * uu] = 0

    rot_mask = np.rot90(mask, k=-1)
    mask_flipped = np.fliplr(rot_mask)
    extended_mask = np.concatenate((rot_mask, mask_flipped), axis=1)
    extended_mask_flipped = np.flipud(extended_mask)
    matrix = np.concatenate((extended_mask, extended_mask_flipped), axis=0)
    
    return matrix

def modifica_linea_obliqua(matrix_size):
    # Creazione di una matrice di zeri
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    threshold_list = [1., 0.7, 0.72, 0.78, 0.85, 0.95]
    center = (matrix_size // 2, matrix_size // 2)

    for i in range(matrix_size):
        for j in range(matrix_size):
            r = distance_from_center(matrix_size, i, j)
            if 2 <= r < matrix_size // 16:
                threshold = threshold_list[1]
            elif matrix_size // 16 <= r < matrix_size // 8:
                threshold = threshold_list[2]
            elif matrix_size // 8 <= r < matrix_size // 4:
                threshold = threshold_list[3]
            elif matrix_size // 4 <= r < matrix_size * 0.4:
                threshold = threshold_list[4]
            elif matrix_size * 0.4 <= r < matrix_size // 2:
                threshold = threshold_list[5]
            else:
                threshold = threshold_list[0]
            if np.random.rand() > threshold:
                # Modifica l'elemento selezionato e i vicini
                modifica_diagonale(matrix, i, j)
                
    matrix[center[0]-2:center[0]+2, center[1]-2:center[1]+2] = 0

    return matrix * build_mask_structure(matrix_size)

def distance_from_center(matrix_size, x_indices, y_indices):
    center = (matrix_size // 2, matrix_size // 2)
    distances = np.sqrt((x_indices - center[0]) ** 2 + (y_indices - center[1]) ** 2)

    return distances

# Funzione per modificare i vicini lungo la diagonale
def modifica_diagonale(matrix, row, col):
    diagonali = [(1, 9), (2, 7), (0, 10), (3, 6)]
    # Imposta l'elemento selezionato a 1
    matrix[row][col] = 1
    # Modifica gli elementi vicini lungo la diagonale
    index = np.random.randint(0, len(diagonali))
    d1, d2 = diagonali[index][0], diagonali[index][1]
    for x in range(d1):
        for y in range(d2):
            new_row = row + x
            new_col = col + y
            if 0 <= new_row < matrix.shape[0] and 0 <= new_col < matrix.shape[1]:
                matrix[new_row][new_col] = 1

if __name__ == "__main__":
    xgan = build_gan(ckpt_path="pgan_5_init_0855.weights.h5")
    print("GAN model built successfully.")

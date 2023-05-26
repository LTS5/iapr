# input: image
# output: image list where every piece is 128x128, also randomized and rotated n*90deg.
import skimage
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rotate
import random

def split_image(image):

    H, W, C = image.shape

    piece_lenght = 128
    num_row_pieces = int(H/piece_lenght)
    num_col_pieces = int(W/piece_lenght)
    num_pieces = num_col_pieces*num_row_pieces

    image_list = np.zeros((num_pieces,piece_lenght,piece_lenght,C), dtype=int)

    # List of possible id:s for pieces in the list
    ids = list(range(num_pieces))

    # Loop over rows and columns
    for row_piece_idx in range(num_row_pieces):
        for col_piece_idx in range(num_col_pieces):
            # Cut out the piece from the image
            piece = image[piece_lenght*row_piece_idx:piece_lenght*(row_piece_idx+1), piece_lenght*col_piece_idx:piece_lenght*(col_piece_idx + 1), :]
            # Rotate it randomly but always in 90 degree intervals
            rot_ang = int(random.randint(0,3) * 90)
            piece = rotate(piece, rot_ang, preserve_range=True).astype(int)

            # Pick randomly in which place to put the image then remove this id from the id list
            id = random.choice(ids)
            ids.remove(id)

            image_list[id] = piece

    return image_list

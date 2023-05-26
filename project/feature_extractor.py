# Input: list of puzzle pieces in an image
# Output: list of featurevectors, one for each puzzle piece
import numpy as np
import matplotlib.pyplot as plt
from feature_reduction import *
from skimage.filters import gabor_kernel

def histogram_features(piece_image, C = 3):
    """
    Extract histogram features from one image.

    Parameters:
    piece_image: (HxWxC) array of the image
    C: (int) number of channels

    Output:
    hist_features: list of histogram features for the piece
    """

    hist_features = []

    # Flatten the height and width dimension
    piece_flat = piece_image.reshape(-1,3)
    
     # Extract features of this channels 
    mean = np.mean(piece_flat, axis = 0)
    std = np.std(piece_flat, axis = 0)
    peak=[]
    for channel in range(C):
        counts = np.bincount(piece_flat[:,channel])
        peak.append(np.argmax(counts))
    
    hist_features.extend(mean)
    hist_features.extend(std)
    hist_features.extend(peak)

    return hist_features

def normalize_image(image):
    """
    Normalize an image.

    Parameters:
    piece_image: (HxWxC) image array

    Output:
    norm_piece_image: (HxWxC) normalized image array
    """
    image = (image - image.mean()) / image.std()

    return image



def gabor_features(piece_image):
    """
    Extract gabor features from one image.

    Parameters:
    piece_image: (HxWxC) array of the image

    Output:
    gabor_features: list of histogram features for the piece
    """

    # Normalize image for better feature extraction
    norm_piece = normalize_image(piece_image)

    # Good code for using gabor filter here https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_gabor.html


    gabor_features = []



    return gabor_features


def normalize_features(features_all):
    """
    Normalize all features.

    Parameters:
    features_all: (NxD) array of all features for all pieces

    Output:
    features: (NxD) normalized array of features for all pieces
    """

    # Mean and standard deviation over all samples
    means = np.mean(features_all, axis=0)
    stds = np.std(features_all, axis=0)

    return (features_all - means)/stds



def extract_features(pieces_list, use_pca = False, d = 3):
    """
    Extract features from the images.

    Parameters:
    pieces_list: (list of arrays) of all pieces in this image
    use_pca: (bool) Use PCA for feature reduction or not
    d: (int) IF PCA, reduce amount of features to this

    Output:
    features: (NxD) array of features for all pieces
    """

    N = len(pieces_list)
    features_all = []
    C = pieces_list[0].shape[2]

    # Extract histogram features from one piece at a time
    for i,piece in enumerate(pieces_list):
        piece_features = []
        # Extract features from histogram
        features_hist = histogram_features(piece, C)
        piece_features.extend(features_hist)

        # Gabor filters for the textures
        features_gabor = gabor_features(piece)
        piece_features.extend(features_gabor)

        # More features we can think of?

        features_all.append(piece_features)
    
    features_all = np.array(features_all)
    print(f'Shape of features: {features_all.shape}')

    # Normalize the features for unbiased clustering/feature reduction
    features_all = normalize_features(features_all)

    # Apply PCA if wanted
    if use_pca:
        print(f'Dimension of features before feature dimension reduction: {features_all.shape}')
        features_all, exvar = PCA(features_all, d)
        print(f'Dimension of features after feature dimension reduction: {features_all.shape}')
        print(f'The total variance explained by the first {d} principal components is {exvar:.3f} %')

    return features_all

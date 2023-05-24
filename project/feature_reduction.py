import numpy as np

def PCA(X, d):
    '''
    Input:
        X: NxD matrix representing our data
        d: Number of principal components to be used to reduce dimensionality
        
    Output:
        Y: Nxd data projected in the principal components' direction
        exvar: explained variance by the principal components
    '''
    # Compute the mean of data
    mean = np.mean(X, axis=0).reshape(1,-1)
    # Center the data with the mean, this needs to be done for PCA to work
    X_tilde = X - mean

    # Create the covariance matrix
    N = X_tilde.shape[0]

    C = 1/N * (X_tilde.T @ X_tilde)

    # Compute the eigenvectors and eigenvalues of the covariance matrix.
    eigvals_full, eigvecs = np.linalg.eigh(C)
    # Choose the top d eigenvalues and corresponding eigenvectors.
    sorted_indicis = np.argsort(eigvals_full)[::-1]
    sorted_n_indicis = sorted_indicis[:d]

    eigvals = eigvals_full[sorted_n_indicis]
    eigvecs = eigvecs[:,sorted_n_indicis]

    W = eigvecs
    eg = eigvals

    # project the data using W
    Y = X_tilde @ W
    
    # Compute the explained variance
    exvar = 100*np.sum(eigvals)/np.sum(eigvals_full)

    return Y, exvar
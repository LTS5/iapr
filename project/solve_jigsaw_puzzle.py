import numpy as np

def solve_jigsaw_puzzle(puzzle_pieces, puzzle_assignments):
    """Merge images in same cluster and keep outliers for themselves

    Args:
        puzzle_pieces np.ndarray: [nPieces, height, width, channels(3)]
        puzzle_assignments: [piece1_cluster, piece2_cluster, ...], where piece1_cluster is int
    """
    outlier_images = []
    solved_puzzles = []

    for cluster_index in range(np.max(puzzle_assignments) + 1):
        cluster_pieces = puzzle_pieces[np.argwhere(puzzle_assignments == cluster_index).ravel()]

        # No pieces - Should never be this
        if cluster_pieces.shape[0] == 0:
            print(f'No pieces in cluster {cluster_index}')
            continue # Next piece
        
        # Len = 1 = outlier
        if cluster_pieces.shape[0] == 1:
            outlier_images.append(cluster_pieces[0])
            continue # Next piece

        # Else merge puzzles together
        solved_puzzles.append(merge_puzzles(cluster_pieces))

    return solved_puzzles, outlier_images

def merge_puzzles(puzzle_pieces):
    return puzzle_pieces[0]
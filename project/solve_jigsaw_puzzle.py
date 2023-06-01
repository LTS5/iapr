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
        cluster_pieces = puzzle_pieces[np.argwhere(np.array(puzzle_assignments) == cluster_index).ravel()]

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
    # res = edge_overlaps(all_pieces, 0)

    # print(f'{res = }')

    # [0, 1, 2, 3] = [top, right, bottom, left]

    index_grid = np.zeros((4,4))

    # Start with one puzzle


    idxs = list(range(12))
    return concat_images(puzzle_pieces, (3,3), idxs)


def concat_images(pieces, grid_size, piece_idxs):
    """concat images

    Args:
        pieces: correctly rotated pieces
        gridsize: Ex: (3, 4) or (rows, columns)
        piece_idx: [idx1, idx2, ] index of piece in grid, 0 is top left and growing to the right

    Returns:
        image: new image with pieces concatinated
    """
    image = np.zeros((grid_size[0]*128, grid_size[1]*128, 3), dtype=np.uint32)

    for piece_idx, piece in zip(piece_idxs, pieces):

        
        x_start, x_end, y_start, y_end = get_start_stop(piece_idx, grid_size)
        image[y_start:y_end, x_start:x_end] = piece

    return image

def get_start_stop(piece_idx, grid_size):
        x_start = piece_idx % grid_size[1]
        y_start = piece_idx // grid_size[1]

        x_end = (x_start+1)*128
        y_end = (y_start+1)*128

        return x_start*128, x_end, y_start*128, y_end
        

def max_edge_overlap(pieces, focus_idx, remove_idx):
    """Compute overlap of all edges in two pieces

    Args:
        piece1 (_type_): _description_
        piece2 (_type_): _description_

    Returns:
        _type_: _description_
    """

    focus_piece = pieces[focus_idx]

    dont_check_indices = remove_idx.append(focus_idx)

    compare_indices = [i for i in range(pieces.shape[0]) if i not in dont_check_indices]
    compare_pieces = pieces[compare_indices]

    overlaps = []
    for i in range(4):
        overlaps.append(
            one_edge_overlap(np.rot90(focus_piece, i)[:,:2,:], compare_pieces)
        )

    # Find max overlap
    max_idx = np.array(overlaps)[:,2].argmax()
    real_idx = compare_indices[max_idx]

    return real_idx, overlaps[max_idx][1]


def one_edge_overlap(edge, pieces):
    """Overlap of one edge to all piece edges

    Args:
        edge: [128, 2]
        pieces: [Npieces, 128, 128]

    Returns:
        
    """
    max_overlap_rot = 0
    max_overlap_val = 9999999
    max_overlap_idx = 0

    # Get max overlap edge for all rotations
    for i in range(3):
        overlap_sums = np.abs(np.rot90(pieces, i)[:,:,:2,:] - edge).sum(axis=(1,2,3))
        rot_overlap_val = np.min(overlap_sums)
        rot_overlap_idx = np.argmin(overlap_sums)

        if rot_overlap_val < max_overlap_val:
            max_overlap_val = rot_overlap_val
            max_overlap_idx = rot_overlap_idx
            max_overlap_rot = i

    return max_overlap_idx, max_overlap_rot, max_overlap_val



if __name__=='__main__':
    from extract_pieces import *
    import os

    def load_input_image(image_index, folder="train", path="data_project"):
        filename = "train_{}.png".format(str(image_index).zfill(2))
        return np.array(Image.open(os.path.join(path,folder,filename)).convert('RGB'))
    
    # images = [load_input_image(i) for i in range(12)]
    # puzzle_outputs = [find_puzzle_pieces(image) for image in images]
    # puzzle_pieces = [out[0] for out in puzzle_outputs]
    # puzzle_masks = [out[1] for out in puzzle_outputs]

    pieces = find_puzzle_pieces(load_input_image(2))[0]

    # # Plot extracted puzzle pieces
    # fig_pieces, axs = plt.subplots((len(pieces)+9) // 10, 10, figsize=(30,10))
    # plt.suptitle(f'Extracted peices: {len(pieces)}', fontsize=60)
    # axs = axs.ravel()

    # # Remove ticks
    # for ax in axs: 
    #     ax.set_xticks([]); ax.set_yticks([])
    #     ax.set_axis_off()

    # # Plot each piece in own subplot
    # for ax, piece in zip(axs, pieces): 
    #     ax.set_axis_on()
    #     ax.imshow(piece)

    # plt.tight_layout()
    # plt.show()

    assigments = [0, 1, 2, 3, 1, 2, 2, 3, 3, 1, 3, 3, 3, 1, 3, 1, 3, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 3]

    solved_puzzles, outlier_images = solve_jigsaw_puzzle(pieces, assigments)

    for solved in solved_puzzles:
        plt.figure()
        plt.imshow(solved)
        plt.show()

    for out in outlier_images:
        plt.figure()
        plt.imshow(out)
        plt.show()

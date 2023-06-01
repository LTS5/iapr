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
            print('Outlier')
            outlier_images.append(cluster_pieces[0])
            continue # Next piece

        # Else merge puzzles together
        print('Cluster')

        # Try to solve
        try:
            solved = merge_puzzles(cluster_pieces)
            print('Solved with jigsaw')
        except:
            # If not able to solve -> Return random
            idxs = list(range(len(cluster_pieces)))

            if len(cluster_pieces) == 9:
                solved = concat_images(puzzle_pieces, (3,3), idxs)
            elif len(cluster_pieces) == 12:
                solved = concat_images(puzzle_pieces, (3,4), idxs)
            elif len(cluster_pieces) == 16:
                solved = concat_images(puzzle_pieces, (4,4), idxs)
            else:
                print(f'Wrong number of puzzle pieces: {len(puzzle_pieces)}')
        
            print('Solved random')
            
        solved_puzzles.append(solved)

    return solved_puzzles, outlier_images

def merge_puzzles(puzzle_pieces):
    # TODO
    # Add check for open sides with length from middle piece max 4
    # Add threshold or max among all free edges



    pieces = puzzle_pieces.copy()
    # [0, 1, 2, 3] = [top, right, bottom, left]

    # Grid to place pieces
    # 7x7 to ensure 4x4 fits when starting in the middle
    index_grid = np.zeros((7,7))-1
    
    # Start with one puzzle
    cur_idx_in_placed_piece = -1
    index_grid[3,3] = 0
    placed_pieces = [0]

    # Do until all pieces are placed
    i = 0
    while len(placed_pieces) < puzzle_pieces.shape[0] and i < 99999: 
        i += 1

        # Update index in placed piece focused on
        cur_idx_in_placed_piece = (cur_idx_in_placed_piece+1) % len(placed_pieces)

        # Set real index
        cur_idx = placed_pieces[cur_idx_in_placed_piece]

        # Check open sides around current piece
        open_sides = open_sides_around(index_grid, cur_idx)

        # No open sides around, go to next placed piece
        if len(open_sides) == 0:
            continue

        # Get the most matching edge from all remaining pieces
        # This checks the current piece against all non-placed piece sides
        piece2_index, piece2_side, focus_piece_side = max_edge_overlap(pieces, cur_idx, sides=open_sides)

        # Insert new piece to grid and rotate it inplace in "pieces"
        insert_piece_to_grid(index_grid, pieces, piece2_index, piece2_side, cur_idx, focus_piece_side)

        # Append newly placed piece
        placed_pieces.append(piece2_index)

    # Get indices
    piece_idxs = index_grid[index_grid != -1]

    # Get shape
    index_grid_bool = np.where(index_grid != -1)
    shape = (index_grid_bool[1].max()-index_grid_bool[1].min(), 
             index_grid_bool[0].max()-index_grid_bool[0].min())


    return concat_images(pieces, shape, piece_idxs)


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
        

def max_edge_overlap(pieces, focus_idx, remove_idx=None, sides=[0,1,2,3]):
    """Compute overlap of all edges in two pieces

    Args:
        pieces:
        focus_idx: int -
        remove_idx: list
        sides:

    Returns:
        index, piece2_side, focus_piece_side, 
    """

    focus_piece = pieces[focus_idx]

    if remove_idx is None:
        dont_check_indices = [focus_idx]
    else:
        dont_check_indices = remove_idx + [focus_idx]

    compare_indices = [i for i in range(pieces.shape[0]) if i not in dont_check_indices]
    compare_pieces = pieces[compare_indices]

    overlaps = []
    for i in sides:
        overlaps.append(
            one_edge_overlap(np.rot90(focus_piece, i)[:,:2,:], compare_pieces)
        )

    # Find max overlap
    max_idx = np.array(overlaps)[:,2].argmax()
    real_idx = compare_indices[max_idx]

    return real_idx, overlaps[max_idx][1], sides[max_idx]


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


def insert_piece_to_grid(grid, pieces, piece2_idx, piece2_side, focus_piece_idx, focus_piece_side):
    """
    Insert new piece into index grid and rotate new piece. Done inplace
    """
    # Get coordinates of focus_piece in grid
    focus_coords = np.where(grid == focus_piece_idx)
    focus_coords = (focus_coords[0][0], focus_coords[1][0])

    # Insert new index into grid and rotate piece2 to match focus_piece
    if focus_piece_side == 0: # Top
        grid[focus_coords[0]-1, focus_coords[1]] = piece2_idx
        rotate_piece_inplace(pieces, piece2_idx, (piece2_side+2)%4)

    elif focus_piece_side == 1: # Right
        grid[focus_coords[0], focus_coords[1]+1] = piece2_idx
        rotate_piece_inplace(pieces, piece2_idx, (piece2_side+1)%4)

    elif focus_piece_side == 2: # Bottom
        grid[focus_coords[0]+1, focus_coords[1]] = piece2_idx
        rotate_piece_inplace(pieces, piece2_idx, piece2_side)

    elif focus_piece_side == 3: # Left
        grid[focus_coords[0], focus_coords[1]-1] = piece2_idx
        rotate_piece_inplace(pieces, piece2_idx, (piece2_side+3)%4)  

def rotate_piece_inplace(pieces, index, rot):
    pieces[index] = np.rot90(pieces[index], rot)

def open_sides_around(grid, focus_index):
    """
    Return sides which are not accupied around index
    """
    focus_coords = np.where(grid == focus_index)
    focus_coords = (focus_coords[0][0], focus_coords[1][0])

    open_sides = []

    # Top
    if grid[focus_coords[0]-1, focus_coords[1]] == -1:
        open_sides.append(0)

    # Right
    if grid[focus_coords[0], focus_coords[1]+1] == -1:
        open_sides.append(1)

    # Bottom
    if grid[focus_coords[0]+1, focus_coords[1]] == -1:
        open_sides.append(2)

    # Left
    if grid[focus_coords[0], focus_coords[1]-1] == -1:
        open_sides.append(3)

    return open_sides

def grid_to_index_array(grid):
    grid[grid != -1]

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

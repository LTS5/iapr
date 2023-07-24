import numpy as np
import matplotlib.pyplot as plt
import cv2
import os 
from PIL import Image

from scipy.signal import peak_prominences
from scipy.ndimage import rotate

def segment_by_hist(image, plot_segmentation=False):
    """
    Segment image based on histogram max peak (assumed to be background)
    
    Parameters:
        image: original image with background and puzzle pieces
        plot_segmentation: plot intermediate results

    Returns: 
        segmented_images: [segmented image 1, segmented image 2, ...]
    """
    # Plot
    if plot_segmentation:
        print('Plots from: segment_by_hist')
        _, axs = plt.subplots(1, 4)

    # Add gray channel
    channels = np.dstack((image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))
    low_thresh, high_thresh = [], []

    # Get histograms for all channels
    hists = [np.histogram(channels[:,:,i].ravel(), bins=255, range=(0,255))[0] for i in range(channels.shape[2])]
    
    # For all channels
    for idx, hist in enumerate(hists):
        # Find peak index = intensity value
        peak = np.argmax(hist)
        
        # Find bases of peak
        _, left_base, right_base = peak_prominences(hist, [peak], wlen=100)

        # Bases will be low and high thresholds
        low_thresh.append(left_base[0])
        high_thresh.append(right_base[0])
        
        # Plot hisogram and lines
        if plot_segmentation:
            axs[idx].hist(channels[:, :, idx].ravel(), bins=255)
            axs[idx].axvline(left_base[0], color='r')
            axs[idx].axvline(right_base[0], color='g')
            axs[idx].axvline(peak, color='k')
            axs[idx].set_xticks([]); axs[idx].set_yticks([])

    # Threshold to binary images
    segmented_images = []
    for i in range(4):
        img = channels[:, :, i]
        img[img < low_thresh[i]] = 0
        img[img > high_thresh[i]] = 0
        segmented_images.append(img.astype(bool).astype(np.uint8))

    # Plot binary images
    if plot_segmentation:
        fig, axs = plt.subplots(1, 4)
        for i in range(len(segmented_images)):
            axs[i].imshow(segmented_images[i])
            axs[i].set_xticks([]); axs[i].set_yticks([])
        plt.show()

    # Return binary images
    return segmented_images

def segment_by_laplacian(image, plot_segmentation=False):
    """
    Segment image based on laplacian values
    
    Parameters:
        image: original image with background and puzzle pieces
        plot_segmentation: plot intermediate results

    Returns: 
        segmented_images: [segmented image 1, segmented image 2, ...]
    """
    # Add gray channel
    channels = np.dstack((image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))

    # Apply Laplacian operator to each channel
    laplacians = [cv2.Laplacian(channels[:,:,i], cv2.CV_64F) for i in range(channels.shape[2])]

    # Segment by threshold
    segmented_images = [(np.abs(lapl) > 10).astype(np.uint8) for lapl in laplacians]
    
    # Morphological close operation to merge points inside squares
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    for i in range(len(segmented_images)):
        segmented_images[i] = cv2.morphologyEx(segmented_images[i], cv2.MORPH_CLOSE, close_kernel)

    # Plot binary images
    if plot_segmentation:
        print('Plots from: segment_by_laplacian')
        fig, axs = plt.subplots(1, 4, figsize=(40,10))
        for i in range(len(laplacians)):
            axs[i].imshow(np.abs(laplacians[i]))
            axs[i].set_xticks([]); axs[i].set_yticks([])
        plt.show()
        fig, axs = plt.subplots(1, 4, figsize=(40,10))
        for i in range(len(segmented_images)):
            axs[i].imshow(segmented_images[i])
            axs[i].set_xticks([]); axs[i].set_yticks([])
        plt.show()

    # Return binary images
    return segmented_images

def find_contours(images):
    """
    Find all contours in segmented images
    
    Parameters:
        images: list where each element is a binary image

    Returns: 
        all_contours: [contour1, contour2, ...]
    """
    # Error handling
    if images is None:
        print('find_contours: Input image is None!')
        return []

    # Find all contours in all images
    all_contours = []
    for i in range(len(images)):
        # Extract contours
        # RETR_EXTERNAL: return all contours without smaller encompasing contours 
        # CHAIN_APPROX_SIMPLE: store only edge points in vertical or horisontal lines
        contours = cv2.findContours(images[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        # Appends since contours is a list
        all_contours += contours

    # Return all extracted contours
    return all_contours

def filter_contours(contours):
    """
    Remove bad contours and create bounding boxes around contours
    
    Parameters:
        contours: All contours

    Returns: 
        good_boxes: [box1, box2, ...] - One bounding box for each good contour
    """
    # Go through all contours
    good_boxes = []
    for c in contours:
        # Get minimal rectangle that encompases contour
        rot_rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rot_rect)
        box = np.int0(box)

        # Only keep boxes of reasonable side lengths
        x = np.append(box[:, 0], box[0, 0])
        y = np.append(box[:, 1], box[0, 1])
        side_lengths = np.sqrt((x[:-1] - x[1:])**2 + (y[:-1] - y[1:])**2)
        
        if (np.abs(side_lengths - 128) > 20).any(): # Skip to next loop, don't add contour
            continue

        # If box already exists - skip duplicate
        duplicate = False
        for gc in good_boxes:
            # Distance from center to center
            center_dist = ((gc.mean(axis=0)-box.mean(axis=0))**2).sum()
            
            if np.sqrt(center_dist) < 30: # Too close
                duplicate = True
                break

        # Add to good_boxes if passed tests and not in array from before
        if not duplicate:
            good_boxes.append(box)

    # Return bounding boxes for all good boxes
    return good_boxes

def refine_boxes(image, boxes, plot_intermediate=False):
    """
    Refine boxes edges and replace with different technique if necessary
    
    Parameters:
        image: original image with background and puzzle pieces
        boxes: coordinates for bounding rectangle vertices for all boxes
        plot_intermediate: plot intermediate results

    Returns: 
        boxes: [box1, box2, ...]
    """
    # For all boxes
    for i in range(len(boxes)):
        # Clip the values to ensure they're inside image (might be -1 ex.)
        boxes[i][:, 0] = np.clip(boxes[i][:, 0], 0, image.shape[1])
        boxes[i][:, 1] = np.clip(boxes[i][:, 1], 0, image.shape[0])

        # Ensure that box is at least 128x128
        box_sides = boxes[i].max(axis=0) - boxes[i].min(axis=0)

        # Try other method if laplacian did not work too good
        if (box_sides < 128).any():
            # Find box from histogram method instead
            segmented_images_hist = segment_by_hist(image, plot_segmentation=plot_intermediate)
            contours_hist = find_contours(segmented_images_hist)
            boxes_hist = filter_contours(contours_hist)

            # Find the box of the same puzzle piece
            for box_hist in boxes_hist:
                center_dist = ((boxes[i].mean(axis=0)-box_hist.mean(axis=0))**2).sum()

                # Exchange old box with new box
                if np.sqrt(center_dist) < 50:
                    if plot_intermediate: # Store to plots new result vs old
                        old_box = boxes[i].copy()

                    boxes[i] = box_hist.copy()

            # Plot old vs new box
            if plot_intermediate:
                _, axs = plt.subplots(1, 2, figsize=(30,10))

                axs[0].imshow(image); axs[0].set_xticks([]); axs[0].set_yticks([])
                axs[1].imshow(image); axs[1].set_xticks([]); axs[1].set_yticks([])

                axs[0].plot(np.append(old_box[:, 0], old_box[0, 0]), np.append(old_box[:, 1], old_box[0, 1]), color='r')
                axs[1].plot(np.append(boxes[i][:, 0], boxes[i][0, 0]), np.append(boxes[i][:, 1], boxes[i][0, 1]), color='r')
                
                
                axs[0].set_title('Old box')
                axs[1].set_title('New box by histogram method')
                plt.show()

    # Return refined boxes
    return boxes

def extract_piece(image, box, plot_intermediate=False):
    """
    Extract pixels inside bounding box
    
    Parameters:
        image: original image with background and puzzle pieces
        box: coordinates for bounding rectangle vertices
        plot_intermediate: plot intermediate results

    Returns: 
        piece: [128, 128, 3] - Extracted puzzle piece
    """
    # Crop to only contain the rotated box (and padding)
    box_min = box.min(axis=0)
    box_max = box.max(axis=0)
    image_cropped = image[box_min[1]:box_max[1], box_min[0]:box_max[0]]

    # Rotate image such that puzzle piece is upright
    dy = box[1, 1] - box[0, 1] # y2 - y1
    dx = box[1, 0] - box[0, 0] # x2 - x1
    angle = np.arctan2(dy, dx) / np.pi * 180

    image_rotated = rotate(image_cropped, angle, reshape=False)

    # Remove padding and extract the centered puzzle piece
    puzzle_piece_size = 128
    pad_x = (image_rotated.shape[1] - puzzle_piece_size) // 2
    pad_y = (image_rotated.shape[0] - puzzle_piece_size) // 2

    piece = image_rotated[pad_y:pad_y+puzzle_piece_size, pad_x:pad_x+puzzle_piece_size]

    # Plot
    # if plot_intermediate:
    if False:
        print('Plots from: extract_piece')
        _, axs = plt.subplots(1, 3, figsize=(30,10))

        # Plot image with box on current piece
        axs[0].imshow(image); axs[0].set_xticks([]); axs[0].set_yticks([])
        axs[0].plot(np.append(box[:, 0], box[0, 0]), np.append(box[:, 1], box[0, 1]), color='r')

        # Plot cropped and rotated image
        axs[1].imshow(image_rotated); axs[1].set_xticks([]); axs[1].set_yticks([])

        # Plot extracted piece
        axs[2].imshow(piece); axs[2].set_xticks([]); axs[2].set_yticks([])
        plt.show()

    # Dobble check that it has the right shape
    assert piece.shape == (128, 128, 3)

    # Return extracted pixels
    return piece

def extract_pieces(image, boxes, plot_intermediate=False):
    """
    Extract pixels inside bounding boxes
    
    Parameters:
        image: original image with background and puzzle pieces
        boxes: coordinates for bounding rectangle vertices for all boxes
        plot_intermediate: plot intermediate results

    Returns: 
        [[128, 128, 3], ...] - Extracted puzzle piece
    """
    return np.array([extract_piece(image, box, plot_intermediate) for box in boxes])

def find_puzzle_pieces(image, plot_results=False, plot_intermediate=False, save_idx=-1):
    """
    Extract individual puzzle pieces from image
    
    Parameters:
        image: original image with background and puzzle pieces
        plot_results: plot bounding boxes and extracted pieces
        plot_intermediate: plot intermediate results from sub-functions

    Returns: 
        puzzle_pieces: [[128, 128, 3], ...] - Extracted puzzle pieces
        full_segmented: [width, height] binary segmented mask
    """
    
    # Segment image by laplacian method
    segmented_images_laplacian = segment_by_laplacian(image, plot_segmentation=plot_intermediate)
    full_segmented = np.array(segmented_images_laplacian).sum(axis=0).astype(np.int0)

    # Get contours from segmented image
    all_contours = find_contours([full_segmented.astype(np.uint8)])

    # Filter out by shape, size and duplicates
    boxes = filter_contours(all_contours)

    # Make sure they have correct size: if not try other method
    boxes = refine_boxes(image, boxes, plot_intermediate=plot_intermediate)

    # Extract pieces from boxes
    puzzle_pieces = extract_pieces(image, boxes, plot_intermediate=plot_intermediate)

    # Plot results
    if plot_results:
        # Plot image with boxes
        fig_boxes, axs = plt.subplots(1, 2)
        plt.title(f'Number of boxes: {len(boxes)}')
        axs[0].imshow(image); axs[0].set_xticks([]); axs[0].set_yticks([])
        axs[1].imshow(image); axs[1].set_xticks([]); axs[1].set_yticks([])

        # Plot all bounding boxes
        for box in boxes:
            axs[1].plot(np.append(box[:, 0], box[0, 0]), np.append(box[:, 1], box[0, 1]), color='r')
        
        # Make it look neat
        axs[1].set_xlim([0,image.shape[1]]); axs[1].set_ylim([image.shape[0],0])
        plt.tight_layout()
        plt.show()

        # Plot extracted puzzle pieces
        fig_pieces, axs = plt.subplots((len(puzzle_pieces)+9) // 10, 10, figsize=(30,10))
        plt.suptitle(f'Extracted peices: {len(puzzle_pieces)}', fontsize=60)
        axs = axs.ravel()

        # Remove ticks
        for ax in axs: 
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_axis_off()

        # Plot each piece in own subplot
        for ax, piece in zip(axs, puzzle_pieces): 
            ax.set_axis_on()
            ax.imshow(piece)

        plt.tight_layout()
        plt.show()

    if save_idx >= 0:
        
        fig_size = (7, 15)
        fig_full = plt.figure(figsize=(10, 5))

        # Original
        ax = plt.subplot(*fig_size, (1,65))
        ax.imshow(image); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'Image {save_idx}')

        # Mask
        ax = plt.subplot(*fig_size, (6,70))
        ax.imshow(full_segmented); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('Mask')

        # Boxes
        ax = plt.subplot(*fig_size, (11,75))
        ax.imshow(image); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'Extracted peices: {len(puzzle_pieces)}')#, fontsize=60)
        for box in boxes:
            ax.plot(np.append(box[:, 0], box[0, 0]), np.append(box[:, 1], box[0, 1]), color='r')

        for idx, piece in enumerate(puzzle_pieces): 
            ax = plt.subplot(*fig_size, 76+idx)
            ax.imshow(piece); ax.set_xticks([]); ax.set_yticks([])

        fig_full.savefig(f"segmentation_data/output_img_{save_idx}.pdf")
        plt.close()

    # Return extracted puzzle pieces and mask
    return puzzle_pieces, full_segmented

if __name__ == '__main__':
    def load_input_image(image_index, folder="train", path="data_project"):
        filename = "train_{}.png".format(str(image_index).zfill(2))
        return np.array(Image.open(os.path.join(path,folder,filename)).convert('RGB'))

    # Test all
    # num_pieces = []
    # import time
    # t1 = time.time()
    # for i in range(12):
    #     test_image = load_input_image(i)
    #     num_pieces.append(len(find_puzzle_pieces(test_image, plot_results=False, plot_intermediate=False, save_idx=i)[0]))
    # t2 = time.time()
    # print(f'Time: {t2 - t1}')

    # answers = np.array([28, 21, 28, 21, 20, 28, 29, 28, 27, 29, 28, 19])
    # print(np.array(num_pieces) - answers)

    # Test one image
    test_image = load_input_image(10)
    segment_by_hist(test_image, plot_segmentation=True)
    puzzle_boxes = find_puzzle_pieces(test_image, plot_results=True, plot_intermediate=True)

    # assign = np.array([np.random.randint(0,5) for _ in range(puzzle_boxes.shape[0])])
    # assign = np.array([np.random.randint(0,5) for _ in range(puzzle_boxes.shape[0])])

    # print(f'{puzzle_boxes.shape = }')
    # print(f'{assign = }')
    # print(f'{np.argwhere(assign==2).ravel() = }')
    # print(f'{puzzle_boxes[np.argwhere(assign==2).ravel()].shape = }')
    # print(f'{puzzle_boxes[np.array([2])].shape = }')
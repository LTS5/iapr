import numpy as np
import matplotlib.pyplot as plt
import cv2

import os 
from PIL import Image
from scipy.signal import peak_prominences

def segment_by_channels(image, plot_segmentation=False):
    thresh_list = []

    # Threshold every image channel
    for i in range(3):
        img = image[:, :, i]
        thresh_new = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh_list.append(thresh_new)

    # Threshold gray image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh_list.append(thresh_gray)

    # Plot segmentations
    if plot_segmentation:
        fig, axs = plt.subplots(1, len(thresh_list))

        for i in range(len(thresh_list)):
            axs[i].imshow(thresh_list[i])
            axs[i].set_xticks([]); axs[i].set_yticks([])

        plt.tight_layout()
        plt.show()

    # Find contours
    all_contours = []
    for i in range(len(thresh_list)):
        contours = cv2.findContours(thresh_list[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # contours = cv2.findContours(thresh_list[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        all_contours += contours

    return all_contours

def segment_by_morph(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (13,13), 2)
    # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # plt.show()

    # fig, axs = plt.subplots(1, 3)
    # axs[0].plot(np.bitwise_and(thresh,thresh_list[2]))
    # plt.show()

    # return

    # plt.imshow(gray)
    # plt.show()
    # plt.imshow(blur)
    # plt.show()
    # plt.imshow(thresh)
    # plt.show()

    # # Two pass dilate with horizontal and vertical kernel
    # struc_size_1 = (10,5)
    # struc_size_2 = (5,10)

    # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, struc_size_1)
    # dilate = cv2.dilate(thresh, horizontal_kernel, iterations=1)
    # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, struc_size_2)
    # dilate = cv2.dilate(dilate, vertical_kernel, iterations=1)

    # # cv2.opening
    # new_im = thresh.copy()
    # close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    # open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    # new_im = cv2.morphologyEx(new_im, cv2.MORPH_CLOSE, close_kernel)
    # new_im = cv2.morphologyEx(new_im, cv2.MORPH_OPEN, open_kernel)
    # full_squares = new_im.copy()

    # new_im = thresh.copy()
    # close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # new_im = cv2.morphologyEx(new_im, cv2.MORPH_CLOSE, close_kernel)
    # new_im = cv2.morphologyEx(new_im, cv2.MORPH_OPEN, open_kernel)
    # sparse_squares = new_im.copy()


    # plt.imshow(full_squares)
    # plt.show()
    # plt.imshow(sparse_squares)
    # plt.show()

    # plt.figure()
    return

def segment_by_hist(image, plot_segmentation=False):
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

    return segmented_images

def segment_by_laplacian(image, plot_segmentation=False):
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
        for i in range(len(segmented_images)):
            axs[i].imshow(segmented_images[i])
            axs[i].set_xticks([]); axs[i].set_yticks([])
        plt.show()

    return segmented_images


def black_white_sharpening(segmented_images, plot_segmentation=False):
    # Morphological operations
    for i in range(len(segmented_images)):
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

        segmented_images[i] = cv2.morphologyEx(segmented_images[i], cv2.MORPH_OPEN, open_kernel)
        segmented_images[i] = cv2.morphologyEx(segmented_images[i], cv2.MORPH_CLOSE, close_kernel)

    # Plot binary images
    if plot_segmentation:
        print('Plots from: black_white_sharpening')

        fig, axs = plt.subplots(1, 4)
        for i in range(len(segmented_images)):
            axs[i].imshow(segmented_images[i])
            axs[i].set_xticks([]); axs[i].set_yticks([])
        plt.show()

    return segmented_images

def find_contours(images):
    # Error handling
    if images is None:
        print('find_contours: Input image is None!')
        return []

    # Find contours all contours in all images
    all_contours = []
    for i in range(len(images)):
        contours = cv2.findContours(images[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        all_contours += contours

    return all_contours

def filter_contours(contours):
    
    good_boxes = []
    for c in contours:
        # Get box coordinates
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

        if not duplicate:
            good_boxes.append(box)

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

from scipy.ndimage import rotate
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
    if plot_intermediate:
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
    return [extract_piece(image, box, plot_intermediate) for box in boxes]

def find_puzzle_pieces(image, plot_results=False, plot_intermediate=False):
    # Segment images by various methods
    # segmented_images_hist = segment_by_hist(image, plot_segmentation=plot_intermediate)
    # segmented_images_bw = black_white_sharpening(segmented_images_hist, plot_segmentation=plot_intermediate)
    segmented_images_laplacian = segment_by_laplacian(image, plot_segmentation=plot_intermediate)


    # Get contours from all methods: + appends results since its normal lists
    # all_contours = find_contours(segmented_images_hist)
    # all_contours += find_contours(segmented_images_bw)
    all_contours = find_contours(segmented_images_laplacian)
    # all_contours += segment_by_channels(image) # += appends results since its normal lists
    # all_contours += find_contours(image) # += appends results since its normal lists

    # Filter out by shape, size and duplicates
    boxes = filter_contours(all_contours)

    # Make sure they have correct size: if not try other method
    boxes = refine_boxes(image, boxes, plot_intermediate=plot_intermediate)

    # Exract pieces
    puzzle_pieces = extract_pieces(image, boxes, plot_intermediate=plot_intermediate)

    # Plot results
    if plot_results:
        fig, axs = plt.subplots(1, 2)
        plt.title(f'Number of boxes: {len(boxes)}')

        axs[0].imshow(image)
        axs[1].imshow(image)

        for box in boxes:
            axs[1].plot(np.append(box[:, 0], box[0, 0]), np.append(box[:, 1], box[0, 1]), color='r')
        
        axs[0].set_xticks([]); axs[0].set_yticks([])
        axs[1].set_xticks([]); axs[1].set_yticks([])
        axs[1].set_xlim([0,image.shape[1]]); axs[1].set_ylim([image.shape[0],0])
        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots((len(puzzle_pieces)+9) // 10, 10, figsize=(30,10))
        plt.suptitle(f'Extracted peices: {len(puzzle_pieces)}', fontsize=60)
        axs = axs.ravel()

        for ax in axs: 
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_axis_off()

        for ax, piece in zip(axs, puzzle_pieces): 
            ax.set_axis_on()
            # print(f'{piece.shape = }')
            ax.imshow(piece)

        plt.tight_layout()
        plt.show()

    return puzzle_pieces

if __name__ == '__main__':
    def load_input_image(image_index, folder="train", path="data_project"):
        filename = "train_{}.png".format(str(image_index).zfill(2))
        return np.array(Image.open(os.path.join(path,folder,filename)).convert('RGB'))

    # # boxes= []
    for i in range(12):
        test_image = load_input_image(i)
        find_puzzle_pieces(test_image, plot_results=False, plot_intermediate=False)

    # answers = np.array([28, 21, 28, 21, 20, 28, 29, 28, 27, 29, 28, 19])
    # print(np.array(boxes) - answers)

    # test_image = load_input_image(2)
    # puzzle_boxes = find_puzzle_pieces(test_image, plot_results=True, plot_intermediate=False)


    # Test on old data
    # for i in range(5):
    #     test_image = load_input_image(i, path="data_project_old")
    #     segment(test_image, plot_results=True, plot_intermediate=False)

    # Notes:
    # laplacian works perfectly on new data
    # laplacian does not work too good on old data
    # Can add hist segmentation for robustness
    # black_white sharpening adds boxes where there should not be boxes - don't use
    # Segment by channels might be unnecessary, only slightly better than hist segmentation
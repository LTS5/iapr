import numpy as np
import matplotlib.pyplot as plt
import cv2

import os 
from PIL import Image

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

from scipy.signal import find_peaks, peak_widths, peak_prominences
def segment_by_hist(image, plot_segmentation=False):
    if plot_segmentation:
        fig, axs = plt.subplots(1, 4)

    # data = np.zeros((image.shape[0], image.shape[1], 4))
    # data[:, :, :3] = image.copy()
    # data[:, :, 3] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = np.dstack((image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))
    low_thresh, high_thresh = [], []

    hists = [np.histogram(data[:,:,i].ravel(), bins=255, range=(0,255))[0] for i in range(4)]
    # hists.append(np.histogram(gray.ravel(), bins=255)[0])
    for idx, hist in enumerate(hists):
        peak = np.argmax(hist)
        # peak_b, props_b = find_peaks(hist_b)
        _, left_base, right_base = peak_prominences(hist, [peak], wlen=100)
        low_thresh.append(left_base[0])
        high_thresh.append(right_base[0])
        
        if plot_segmentation:
            axs[idx].hist(data[:, :, idx].ravel(), bins=255)
            axs[idx].axvline(left_base[0], color='r')
            axs[idx].axvline(right_base[0], color='g')
            axs[idx].axvline(peak, color='k')
            axs[idx].set_xticks([]); axs[idx].set_yticks([])

    # Threshold to binary images
    segmented_images = []
    for i in range(4):
        img = data[:, :, i]
        img[img < low_thresh[i]] = 0
        img[img > high_thresh[i]] = 0
        segmented_images.append(img.astype(bool).astype(np.uint8)) # +0 to convert to int 0 and 1 matrix

    # Plot binary images
    if plot_segmentation:
        plt.show()

        fig, axs = plt.subplots(1, 4)
        for i in range(len(segmented_images)):
            axs[i].imshow(segmented_images[i])
            axs[i].set_xticks([]); axs[i].set_yticks([])
        plt.show()
    
    # Find contours
    all_contours = []
    for i in range(len(segmented_images)):
        contours = cv2.findContours(segmented_images[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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

def segment(image, plot_results=False):
    # Try hist
    # Try hue, saturation, value channels

    # Get contours from all methods
    all_contours = segment_by_channels(image)
    all_contours += segment_by_hist(image)

    # Filter out by shape, size and duplicates
    boxes = filter_contours(all_contours)

    # Plot results
    if plot_results:
        fig, axs = plt.subplots(1, 2)

        axs[0].imshow(image)
        axs[1].imshow(image)

        for box in boxes:
            axs[1].plot(np.append(box[:, 0], box[0, 0]), np.append(box[:, 1], box[0, 1]), color='r')
        
        plt.suptitle(f'Number of boxes: {len(boxes)}')
        axs[0].set_xticks([]); axs[0].set_yticks([])
        axs[1].set_xticks([]); axs[1].set_yticks([])
        plt.tight_layout()
        plt.show()

    # return 

if __name__ == '__main__':
    def load_input_image(image_index, folder="train", path="data_project"):
        filename = "train_{}.png".format(str(image_index).zfill(2))
        return np.array(Image.open(os.path.join(path,folder,filename)).convert('RGB'))

    for i in range(12):
        test_image = load_input_image(i)
        puzzle_boxes = segment(test_image, plot_results=True)

    # test_image = load_input_image(2)
    # puzzle_boxes = segment(test_image, plot_results=True)

    # Old code
    # def black_white_sharpening(segmented_images, plot_segmentation=False):
        # # Morphological operations
        # for i in range(len(segmented_images)):
        #     close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        #     open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

        #     segmented_images[i] = cv2.morphologyEx(segmented_images[i], cv2.MORPH_OPEN, open_kernel)
        #     segmented_images[i] = cv2.morphologyEx(segmented_images[i], cv2.MORPH_CLOSE, close_kernel)

        # # Plot binary images
        # if plot_segmentation:
        #     print('Plots from: black_white_sharpening')

        #     fig, axs = plt.subplots(1, 4)
        #     for i in range(len(segmented_images)):
        #         axs[i].imshow(segmented_images[i])
        #         axs[i].set_xticks([]); axs[i].set_yticks([])
        #     plt.show()

        # return segmented_images


    # def segment_by_channels(image, plot_segmentation=False):
    # thresh_list = []

    # # Threshold every image channel
    # for i in range(3):
    #     img = image[:, :, i]
    #     thresh_new = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #     thresh_list.append(thresh_new)

    # # Threshold gray image
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # thresh_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # thresh_list.append(thresh_gray)

    # # Plot segmentations
    # if plot_segmentation:
    #     fig, axs = plt.subplots(1, len(thresh_list))

    #     for i in range(len(thresh_list)):
    #         axs[i].imshow(thresh_list[i])
    #         axs[i].set_xticks([]); axs[i].set_yticks([])

    #     plt.tight_layout()
    #     plt.show()

    # # Find contours
    # all_contours = []
    # for i in range(len(thresh_list)):
    #     contours = cv2.findContours(thresh_list[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #     # contours = cv2.findContours(thresh_list[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     contours = contours[0] if len(contours) == 2 else contours[1]

    #     all_contours += contours

    # return all_contours

    # def segment_by_morph(image):
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
    # return
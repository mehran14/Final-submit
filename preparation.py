import os
import urllib
from io import BytesIO
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import requests
from patchify import patchify
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
import cv2


def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image


def show_patch_images(patched_images):
    num_patches = len(patched_images)

    # Calculate the grid size for subplots
    num_cols = 4
    num_rows = (num_patches + num_cols - 1) // num_cols

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

    # Flatten the axes array if it's not already 2D
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    # Iterate over the patches and display them
    for i in range(num_patches):
        patch = patched_images[i]

        # Display the patch in the corresponding subplot
        ax = axes[i // num_cols, i % num_cols]
        ax.imshow(patch)
        ax.axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.show()


def load_solar_panel_dataset_POV03(image_dir, patch_size):

    image_dataset = []
    mask_dataset = []

    for dirname, _, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.endswith(".bmp"):
                if not filename.endswith('_label.bmp'):
                    # Load input images
                    image = cv2.imread(os.path.join(dirname, filename), 1) # Read each image as BGR
                    SIZE_X = (image.shape[1]//patch_size)*patch_size # The nearest size divisible by our patch size
                    SIZE_Y = (image.shape[0]//patch_size)*patch_size # The nearest size divisible by our patch size
                    image = Image.fromarray(image)
                    image = image.crop((0 ,0, SIZE_X, SIZE_Y))  # Crop from the top left corner
                    image = np.array(image)

                    # Extract patches from each image
                    print("patchifying image:", dirname + "/" + filename)
                    patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)  # Step=256 for 256 patches means no overlap

                    for i in range(patches_img.shape[0]):
                        for j in range(patches_img.shape[1]):
                            single_patch_img = patches_img[i,j,:,:]
                            # Use minmaxscaler instead of just dividing by 255.
                            single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                            #single_patch_img = (single_patch_img.astype('float32')) / 255.
                            single_patch_img = single_patch_img[0]  # Drop the extra unnecessary dimension that patchify adds.
                            image_dataset.append(single_patch_img)

                    # Load labeled images
                    mask = cv2.imread(os.path.join(dirname, filename[:-4] + '_label.bmp'), 1)
                    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                    SIZE_X = (mask.shape[1]//patch_size)*patch_size  # The nearest size divisible by our patch size
                    SIZE_Y = (mask.shape[0]//patch_size)*patch_size  # The nearest size divisible by our patch size
                    mask = Image.fromarray(mask)
                    mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from the top left corner
                    #mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                    mask = np.array(mask)

                    #Extract patches from each image
                    print("patchifying mask:", dirname + "/" + filename)
                    patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap

                    for i in range(patches_mask.shape[0]):
                        for j in range(patches_mask.shape[1]):
                            single_patch_mask = patches_mask[i,j,:,:]
                            #single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                            single_patch_mask = single_patch_mask[0] # Drop the extra unnecessary dimension that patchify adds.
                            mask_dataset.append(single_patch_mask)

    return image_dataset, mask_dataset


def load_solar_panel_dataset(image_dir, patch_size):

    image_dataset = []
    mask_dataset = []

    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            if not filename.endswith('_label.png'):
                # Load input images
                image = cv2.imread(os.path.join(image_dir, filename), 1) # Read each image as BGR
                SIZE_X = (image.shape[1]//patch_size)*patch_size # The nearest size divisible by our patch size
                SIZE_Y = (image.shape[0]//patch_size)*patch_size # The nearest size divisible by our patch size
                image = Image.fromarray(image)
                image = image.crop((0 ,0, SIZE_X, SIZE_Y))  # Crop from the top left corner
                image = np.array(image)

                # Extract patches from each image
                print("patchifying image:", image_dir + "/" + filename)
                patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)  # Step=256 for 256 patches means no overlap

                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):

                        single_patch_img = patches_img[i,j,:,:]

                        # Use minmaxscaler instead of just dividing by 255.
                        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                        #single_patch_img = (single_patch_img.astype('float32')) / 255.
                        single_patch_img = single_patch_img[0] # Drop the extra unnecessary dimension that patchify adds.
                        image_dataset.append(single_patch_img)


                # Load labeled images
                mask = cv2.imread(os.path.join(image_dir, filename[:-4] + '_label.png'), 1)
                mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                SIZE_X = (mask.shape[1]//patch_size)*patch_size #The nearest size divisible by our patch size
                SIZE_Y = (mask.shape[0]//patch_size)*patch_size #The nearest size divisible by our patch size
                mask = Image.fromarray(mask)
                mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from the top left corner
                #mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                mask = np.array(mask)

                #Extract patches from each image
                print("patchifying mask:", image_dir + "/" + filename)
                patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap

                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        single_patch_mask = patches_mask[i,j,:,:]
                        #single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                        single_patch_mask = single_patch_mask[0] #Drop the extra unnecessary dimension that patchify adds.
                        mask_dataset.append(single_patch_mask)

    return image_dataset, mask_dataset


def load_solar_panel_dataset_PV08(image_dir, patch_size):

    image_dataset = []
    mask_dataset = []

    for filename in os.listdir(image_dir):
        if filename.endswith(".bmp"):
            if not filename.endswith('_label.bmp'):
                # Load input images
                image = cv2.imread(os.path.join(image_dir, filename), 1) # Read each image as BGR
                SIZE_X = (image.shape[1]//patch_size)*patch_size # The nearest size divisible by our patch size
                SIZE_Y = (image.shape[0]//patch_size)*patch_size # The nearest size divisible by our patch size
                image = Image.fromarray(image)
                image = image.crop((0 ,0, SIZE_X, SIZE_Y))  # Crop from the top left corner
                image = np.array(image)

                # Extract patches from each image
                print("patchifying image:", image_dir + "/" + filename)
                patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)  # Step=256 for 256 patches means no overlap

                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):

                        single_patch_img = patches_img[i,j,:,:]

                        # Use minmaxscaler instead of just dividing by 255.
                        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                        #single_patch_img = (single_patch_img.astype('float32')) / 255.
                        single_patch_img = single_patch_img[0] # Drop the extra unnecessary dimension that patchify adds.
                        image_dataset.append(single_patch_img)


                # Load labeled images
                mask = cv2.imread(os.path.join(image_dir, filename[:-4] + '_label.bmp'), 1)
                mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                SIZE_X = (mask.shape[1]//patch_size)*patch_size #The nearest size divisible by our patch size
                SIZE_Y = (mask.shape[0]//patch_size)*patch_size #The nearest size divisible by our patch size
                mask = Image.fromarray(mask)
                mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from the top left corner
                #mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                mask = np.array(mask)

                #Extract patches from each image
                print("patchifying mask:", image_dir + "/" + filename)
                patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap

                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        single_patch_mask = patches_mask[i,j,:,:]
                        #single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                        single_patch_mask = single_patch_mask[0] #Drop the extra unnecessary dimension that patchify adds.
                        mask_dataset.append(single_patch_mask)

    return image_dataset, mask_dataset


def fit_image_model(image):
    try:

        # Convert the image to a numpy array if it is not already
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Convert an image data type to compatible format for OpenCV
        image = image.astype(np.uint8)

        # Resize the image to (256, 256)
        resized_image = cv2.resize(image, (256, 256))

        # If the image has a single channel, convert it to three channels
        if len(resized_image.shape) == 2:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)

        # If the image has an alpha channel, remove it
        if resized_image.shape[2] == 4:
            resized_image = resized_image[:, :, :3]

        # Ensure the image has 3 channels
        if resized_image.shape[2] != 3:
            raise ValueError("Invalid image format. Expected 3 channels.")

        return resized_image
    except Exception as e:
        print("Error: ", e)
        return None


def plot_image(image):
    try:
        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Plot the image
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print("Error: ", e)


def random_image_mask_show(images, masks):
    image_dataset = images
    mask_dataset = masks
    patch_size = 256

    image_number1 = random.randint(0, len(image_dataset))
    image_number2 = random.randint(0, len(image_dataset))
    image_number3 = random.randint(0, len(image_dataset))
    image_number4 = random.randint(0, len(image_dataset))

    plt.figure(figsize=(14, 7))

    plt.subplot(241)
    plt.title('Random Image 1')
    plt.imshow(np.reshape(image_dataset[image_number1], (patch_size, patch_size, 3)))
    plt.subplot(242)
    plt.title('Random Mask 1')
    plt.imshow(np.reshape(mask_dataset[image_number1], (patch_size, patch_size, 3)))

    plt.subplot(243)
    plt.title('Random Image 2')
    plt.imshow(np.reshape(image_dataset[image_number2], (patch_size, patch_size, 3)))
    plt.subplot(244)
    plt.title('Random Mask 2')
    plt.imshow(np.reshape(mask_dataset[image_number2], (patch_size, patch_size, 3)))

    plt.subplot(245)
    plt.title('Random Image 3')
    plt.imshow(np.reshape(image_dataset[image_number3], (patch_size, patch_size, 3)))
    plt.subplot(246)
    plt.title('Random Mask 3')
    plt.imshow(np.reshape(mask_dataset[image_number3], (patch_size, patch_size, 3)))

    plt.subplot(247)
    plt.title('Random Image 4')
    plt.imshow(np.reshape(image_dataset[image_number4], (patch_size, patch_size, 3)))
    plt.subplot(248)
    plt.title('Random Mask 4')
    plt.imshow(np.reshape(mask_dataset[image_number4], (patch_size, patch_size, 3)))

    plt.tight_layout

    return plt


def rand_predict_result_dataset(model, X_dataset, y_dataset):
    X_test = X_dataset
    y_test_argmax = y_dataset
    model = model
    test_img_number = random.randint(0, len(X_test))
    test_img = X_test[test_img_number]
    ground_truth = y_test_argmax[test_img_number]
    # test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input = np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input))
    predicted_img = np.argmax(prediction, axis=3)[0, :, :]

    plt.figure(figsize=(12, 7))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img)
    plt.subplot(232)
    plt.title('Label')
    plt.imshow(ground_truth)
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img)

    plt.tight_layout
    return plt


def unique_img_hex_val_colors(image):
    hex_val_colors = []
    pixels = image

    for y in range(image.shape[1]):  # this row
        for x in range(image.shape[0]):  # and this row was exchanged
            r, g, b = pixels[x, y]
            # in case your image has an alpha channel
            # r, g, b, a = pixels[x, y]
            hex_val_colors.append(f"#{r:02x}{g:02x}{b:02x}")

    unique_colors = np.unique(hex_val_colors)
    return unique_colors


def rgb_to_hex_dataset(dataset):
    """
    RGB to HEX: (Hexadecimal --> base 16)
    This number divided by sixteen (integer divisions; ignoring any remainder) gives
    the first hexadecimal digit (between 0 and F, where the letters A to F represent
    the numbers 10 to 15). The remainder gives the second hexadecimal digit.
    0-9 --> 0-9
    10-15 --> A-F

    Example: RGB --> R=201, G=, B=

    R = 201/16 = 12 with the remainder of 9. So hex code for R is C9 (remember C=12)

    Calculating RGB from HEX: #3C1098
    3C = 3*16 + 12 = 60
    10 = 1*16 + 0 = 16
    98 = 9*16 + 8 = 152

    :param dataset: an RGB image dataset
    :return: a dataset transferred to Hex
    """

    hex_dataset = dataset

    for img_count in range(len(hex_dataset)):
        pixels = hex_dataset[img_count]

        for y in range(hex_dataset[img_count].shape[1]):  # this row
            for x in range(hex_dataset[img_count].shape[0]):  # and this row was exchanged
                r, g, b = pixels[x, y]
                color = f"#{r:02x}{g:02x}{b:02x}"
                if not color == '#000000':
                    hex_dataset[img_count, x, y, :] = (211, 211, 211)

    return hex_dataset


def rgb_to_2D_solar_panel(input_label):
    """
    Supply our label masks as input in RGB format.
    Replace pixels with specific RGB values ...
    """
    solar_panels = '#d3d3d3'.lstrip('#')
    solar_panels = np.array(tuple(int(solar_panels[i:i + 2], 16) for i in (0, 2, 4)))

    land = '#000000'.lstrip('#')
    land = np.array(tuple(int(land[i:i + 2], 16) for i in (0, 2, 4)))

    label_seg = np.zeros(input_label.shape,dtype=np.float32)
    label_seg [np.all(input_label == solar_panels, axis=-1)] = 1
    label_seg [np.all(input_label == land, axis=-1)] = 0

    label_seg = label_seg[:,:,0]  # Just take the first channel, no need for all 3 channels

    return label_seg


def rgb_to_2D_dataset(dataset):
    labels = []

    for i in range(dataset.shape[0]):
        label = rgb_to_2D_solar_panel(dataset[i])
        labels.append(label)

    return labels


def predict_image(model, image):
    try:
        test_img = image
        test_img_input = np.expand_dims(test_img, 0)
        prediction = (model.predict(test_img_input))
        predicted_img = np.argmax(prediction, axis=3)[0, :, :]

        # Get the predicted class label and score
        predicted_class = np.argmax(prediction)
        score = prediction[0, predicted_class]

        # Show the figure
        plt.figure(figsize=(12, 7))
        plt.subplot(121)
        plt.title('Testing Image')
        plt.imshow(test_img)
        plt.subplot(122)
        plt.title('Prediction on test image')
        plt.imshow(predicted_img)

        plt.tight_layout()
        plt.show()

        # Return the predicted score
        return score

    except Exception as e:
        print("Error: ", e)
        return None
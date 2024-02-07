import PIL.Image
import cv2
import numpy
import numpy as np
import os
import math
import skimage
import scipy
import dlib
from PIL import Image, ImageOps
from PIL import ImageDraw

'''
Name: Rhiannon Barber
Date: January 25, 2024

This file contains the code for augmenting the facial features (nose, and eyes currently as of 1/25).

The augmentation is done to a hardcoded scale value, that needs to be changed to work with the std
deviation found in the Facial_Landmarking_Math.py file. 

As of 1/25 I have successfully blurred in the nose region for both larger and smaller scale values. 

Currently working on the eyes, though the code is not included in this file since it is very messy and not 
working correctly.  

In line comments have more detail on how certain things are working. 
'''


# This function constructs a file path for the specific directory.
def get_dir(filename):
    if "Augmentation_Project" in os.getcwd():
        filename = os.path.join(os.getcwd(), filename)
    else:
        filename = os.path.join(os.getcwd(), ("Augmentation_Project/" + filename))
    return filename


'''
This function loads the land markings found from the .npy file
returns the x,y coordinates from the data with are col 4 & 5.
'''


def load_feature_landmarks(facial_features, image_id, feature_to_int, feature_name):
    feature_landmarks = facial_features[
        (facial_features[:, 0] == image_id) & (facial_features[:, 1] == feature_to_int[feature_name])]
    return feature_landmarks[:, 3:5]


'''
 This function is responsible for resizing the nose.
 Gets the coordinates from the .npy file, and finds the feature named 'nose'
 returns the resized nose, with a margin factor to capture the nostrils
 region is overlaid on the image.
'''


def resize_nose(img, facial_features, image_id, feature_to_int, scale_factor):
    nose_landmarks = load_feature_landmarks(facial_features, image_id, feature_to_int, 'nose')
    return resize_and_overlay_feature(img, nose_landmarks, scale_factor, width_margin_factor=0.6,
                                      height_margin_factor=0.7)


'''
Similar to resize nose, but for eyes.
returns the resized eyes, with margin to capture eye lashes.
region is overlaid on the image.
'''


def resize_eyes(img, facial_features, image_id, feature_to_int, scale_factor):
    eye_landmarks = load_feature_landmarks(facial_features, image_id, feature_to_int, 'eyes')
    return resize_and_overlay_feature(img, eye_landmarks, scale_factor, width_margin_factor=0.25,
                                      height_margin_factor=0.4)


'''
This function is what is returned in the resize method. 
It's responsible for resizing the feature points, based on the scale factor, and the margin points. 
it is then overlaid onto the original image, and the 
'''


def resize_and_overlay_feature(img, feature_points, scale_factor, width_margin_factor, height_margin_factor):
    # Calculates the bounding box for the region, based on margin and landmarks.
    x_min, y_min = np.min(feature_points, axis=0).astype(int)
    x_max, y_max = np.max(feature_points, axis=0).astype(int)
    width_margin = int((x_max - x_min) * width_margin_factor)
    height_margin = int((y_max - y_min) * height_margin_factor)
    x_min = max(x_min - width_margin, 0)
    y_min = max(y_min - height_margin, 0)
    x_max = min(x_max + width_margin, img.shape[1])
    y_max = min(y_max + height_margin, img.shape[0])
    feature_region = img[y_min:y_max, x_min:x_max]
    feature_width = x_max - x_min
    feature_height = y_max - y_min
    scaled_width = int(feature_width * scale_factor)
    scaled_height = int(feature_height * scale_factor)

    # cv2.resize takes the image, the scale to resize, and the interpolation: I chose linear since it was the
    # simplest approach to scaling the region with good results.
    scaled_feature = cv2.resize(feature_region, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

    # Calculates the overlay starting horizontal margin for where the resized feature should be placed.
    # Discarding the fractional part to ensure whole integers for coordinates.
    overlay_start_x = x_min + (feature_width - scaled_width) // 2

    # Calculates the overlay starting vertical margin for where the resized feature should be placed.
    # Discarding the fractional part to ensure whole integers for coordinates.
    overlay_start_y = y_min + (feature_height - scaled_height) // 2
    img[overlay_start_y:overlay_start_y + scaled_height, overlay_start_x:overlay_start_x + scaled_width] = scaled_feature
    return img


'''
This function creates a mask where the area around the nose is white (255), to indicate the ROI. 
The rest of the mask is black, it can then be used in the blending.
The mask uses the land mark points, and margin to define the region of the mask. 
'''


def create_nose_mask(img, feature_points, width_margin_factor, height_margin_factor):
    mask = np.zeros_like(img)
    x_min, y_min = np.min(feature_points, axis=0).astype(int)
    x_max, y_max = np.max(feature_points, axis=0).astype(int)
    width_margin = int((x_max - x_min) * width_margin_factor)
    height_margin = int((y_max - y_min) * height_margin_factor)
    x_min = max(x_min - width_margin, 0)
    y_min = max(y_min - height_margin, 0)
    x_max = min(x_max + width_margin, img.shape[1])
    y_max = min(y_max + height_margin, img.shape[0])
    mask[y_min:y_max, x_min:x_max] = 255
    return mask


def create_eye_mask(img, feature_points, width_margin_factor, height_margin_factor):
    mask = np.zeros_like(img)
    x_min, y_min = np.min(feature_points, axis=0).astype(int)
    x_max, y_max = np.max(feature_points, axis=0).astype(int)
    width_margin = int((x_max - x_min) * width_margin_factor)
    height_margin = int((y_max - y_min) * height_margin_factor)
    x_min = max(x_min - width_margin, 0)
    y_min = max(y_min - height_margin, 0)
    x_max = min(x_max + width_margin, img.shape[1])
    y_max = min(y_max + height_margin, img.shape[0])
    mask[y_min:y_max, x_min:x_max] = 255
    return mask


def main():
    facial_features = np.load(get_dir('facial_features.npy'))
    feature_to_int = {'jawline': 0, 'eyebrows': 1, 'nose': 2, 'eyes': 3, 'lips': 4}
    image_directory = get_dir('original_images')
    augmented_directory = get_dir('augmented_images')
    nose_scale_factor = 1.25
    eye_scale_factor = 0.80

    if not os.path.exists(augmented_directory):
        os.makedirs(augmented_directory)

    image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if
                   filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_path in image_paths:
        img = cv2.imread(img_path)
        image_id = image_paths.index(img_path)
        original_img = img.copy()
        img = resize_nose(img, facial_features, image_id, feature_to_int, nose_scale_factor)
        nose_landmarks = load_feature_landmarks(facial_features, image_id, feature_to_int, 'nose')
        # the margins here for the mask affect where the bounds of the box are.
        nose_mask = create_nose_mask(original_img, nose_landmarks, width_margin_factor=0.4, height_margin_factor=0.1)

        # blurs the nose region, the kernel size changes the blur amount
        # ksize (0,0)  -> (width, height)
        # sigmaX -> 21 -> is the standard deviation of the kernel.
        # higher sigmaX = more blur. 21 showed good results. 24+ had some loss of detail.
        nose_mask_blurred = cv2.GaussianBlur(nose_mask, (0, 0), 21)

        # (nose_mask_blurred / 255) -> normalizes values to be between 0 & 1
        # areas are either 255 (white) rep by 1 or 0 (black) rep by 0
        # img * (nose_mask_blurred / 255) -> White areas are kept the same, and black areas are blurred.
        # original_img * (1 - (nose_mask_blurred / 255) -> Does above but in inverse.
        # white is now rep by 0, black is rep by 1
        # areas that are 0 (black) are now kept, and areas that are white are blurred.
        # adding them together gives the seamless blend of the images; so the nose is augmented and now blended in.
        img = img * (nose_mask_blurred / 255) + original_img * (1 - (nose_mask_blurred / 255))

        # saves the image with augmented_filename.jpeg for our way of telling the images apart, and ease of use in UI.
        # saved to the augmented_images dir.
        augmented_img_name = f"augmented_{os.path.basename(img_path)}"
        augmented_img_path = os.path.join(augmented_directory, augmented_img_name)

        # START TO EYE AUGMENTATION #
        # Get landmarks for eyes and other facial features
        eye_landmarks = load_feature_landmarks(facial_features, image_id, feature_to_int, 'eyes')
        eyebrow_landmarks = load_feature_landmarks(facial_features, image_id, feature_to_int, 'eyebrows')
        face_landmarks = load_feature_landmarks(facial_features, image_id, feature_to_int, 'jawline')
        nose_landmarks = load_feature_landmarks(facial_features, image_id, feature_to_int, 'nose')

        # Delete eye_landmarks to make shape of numpy arrays the same
        left_eye, right_eye = numpy.array_split(eye_landmarks, 2)  # Delete the second outward landmark
        left_brow, right_brow = numpy.array_split(eyebrow_landmarks, 2)
        l_eye, r_eye = left_eye.copy(), right_eye.copy()

        # Delete outermost eyebrow landmark to
        # make upper eye and eyebrow landmarks number the same
        left_brow = numpy.delete(left_brow, 3, 0)
        right_brow = numpy.delete(right_brow, 1, 0)

        # Delete bottom two eye landmarks
        left_eye = numpy.delete(left_eye, 4, 0)
        left_eye = numpy.delete(left_eye, 4, 0)
        right_eye = numpy.delete(right_eye, 4, 0)
        right_eye = numpy.delete(right_eye, 4, 0)

        # Expand eye_landmarks by half the distance from eyes to eyebrows
        result1 = (left_eye - left_brow)
        result2 = (right_eye - right_brow)
        result1[0, 0] = -(face_landmarks[0, 0] - left_eye[0, 0])/2  # make horizontal mask wider
        result2[3, 0] = -((face_landmarks[16, 0] - right_eye[3, 0])/2)  # make horizontal mask wider
        l_lower_expand = (result1[3, 1])  # Get the largest distance from results
        r_lower_expand = (result1[3, 1])   # Get the largest distance from results
        result1[3, 0] = (left_eye[3, 0] - nose_landmarks[0, 0])/2   # Expand eye near nose horizontal
        result2[0, 0] = -(nose_landmarks[0, 0] - right_eye[0, 0])/2  # Expand eye near nose horizontal
        result1, result2 = result1 / 2, result2 / 2
        # Insert bottom eye landmarks and expand downward
        # result1 = numpy.append(result1, [[-(face_landmarks[0, 0] - left_eye[0, 0])/2, -l_lower_expand]], axis=0)
        result1 = numpy.append(result1, [[-(face_landmarks[0, 0] - left_eye[0, 0])/2, -l_lower_expand/2]], axis=0)
        result1 = numpy.insert(result1, 4, [-result1[0, 0], -l_lower_expand], axis=0)
        result2 = numpy.append(result2, [[result2[3, 1], -r_lower_expand]], axis=0)
        result2 = numpy.insert(result2, 4, [-(face_landmarks[16, 0] - right_eye[3, 0])/2, -r_lower_expand/2], axis=0)
        # Calculate expanded result into eye_landmarks
        eye_landmarks1, eye_landmarks2 = l_eye - result1,  r_eye - result2

        # mask 1, makes mask for reduced eye
        mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(eye_landmarks1), (1.0, 1.0, 1.0))
        cv2.fillConvexPoly(mask, np.int32(eye_landmarks2), (1.0, 1.0, 1.0))
        mask = 255*np.uint8(mask)
        # mask 2, makes mask for blurred background of eye
        eye_landmarks1[0, 0] = eye_landmarks1[0, 0]+((face_landmarks[0, 0] - eye_landmarks1[0, 0])/2)  # make horizontal mask wider
        eye_landmarks2[3, 0] = eye_landmarks2[3, 0]+((face_landmarks[16, 0] - eye_landmarks2[3, 0])/2)  # make horizontal mask wider
        eye_landmarks1[5, 0] = eye_landmarks1[5, 0]-(face_landmarks[16, 0] - eye_landmarks2[3, 0])  # make horizontal mask wider
        eye_landmarks2[4, 0] = eye_landmarks2[4, 0]+(face_landmarks[16, 0] - eye_landmarks2[3, 0])  # make horizontal mask wider
        # print(eye_landmarks1[0, 0])
        mask2 = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask2, np.int32(eye_landmarks1), (1.0, 1.0, 1.0))
        cv2.fillConvexPoly(mask2, np.int32(eye_landmarks2), (1.0, 1.0, 1.0))
        mask2 = 255*np.uint8(mask2)

        # Apply close operation to improve mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        blur = (3*int(result1[0, 0]))
        # Blur the mask to obtain natural result
        if not(blur % 2):
            blur = blur + 1
        mask = cv2.GaussianBlur(mask, (blur, blur), 99)
        mask2 = cv2.GaussianBlur(mask2, (99, 99), 32)
        # cv2.imshow("", mask2)
        # cv2.waitKey(0)

        # Calculate inverse mask
        inverse_mask = cv2.bitwise_not(mask)
        inverse_mask2 = cv2.bitwise_not(mask2)

        # Convert masks to float to perform blending
        mask = mask.astype(float)/255
        inverse_mask = inverse_mask.astype(float)/255
        mask2 = mask2.astype(float)/255
        inverse_mask2 = inverse_mask2.astype(float)/255

        # Apply color mapping for the eyes
        img = img.astype(np.uint8)
        face_width = face_landmarks[0, 1]
        eyebrow_height = eyebrow_landmarks[3, 1]
        h, w, c = img.shape
        height_reduced = int((h-eyebrow_height)/32)
        width_reduced = int((w-face_width)/8)
        eye = cv2.copyMakeBorder(img, height_reduced, height_reduced*3, width_reduced, width_reduced, cv2.BORDER_CONSTANT)
        eye = cv2.resize(eye, (w, h))
        face = img.copy()
        face = cv2.resize(face, (w*2, h*2))
        face = cv2.copyMakeBorder(face, 0, 0, int(width_reduced*0.3), int(width_reduced*0.3), cv2.BORDER_CONSTANT)
        face = cv2.resize(face, (w, h))
        face = cv2.GaussianBlur(face, (0, 0), 8)

        # Convert lips and face to 0-1 range
        eye = eye.astype(float)/255
        face = face.astype(float)/255
        head = img.astype(float)/255

        # Multiply eyes and face by the mask
        just_eye = cv2.multiply(mask2, face)
        just_face = cv2.multiply(inverse_mask2, head)
        result = just_face + just_eye  # Add face and eye

        # Multiply eyes and face by the mask2
        just_eye = cv2.multiply(mask, eye)
        just_face = cv2.multiply(inverse_mask, result)
        result = just_face + just_eye  # Add face and eye
        # Show result
        result = cv2.normalize(result, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(augmented_img_path, result)


if __name__ == "__main__":
    main()

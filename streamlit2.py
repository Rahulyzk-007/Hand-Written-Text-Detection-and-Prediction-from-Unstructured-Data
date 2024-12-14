#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import joblib
import pytesseract
import cv2
import matplotlib.pyplot as plt
import pickle
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os


# In[ ]:


import streamlit as st


# In[ ]:


mj=joblib.load('/Users/rahul/Documents/SummerInternship/act_model__joblib_10_epochs')
cnn=joblib.load('/Users/rahul/Documents/SummerInternship/HWvsDT_main_model')


# In[ ]:


char_set={'M', "'", '/', ')', 'm', 'q', 'k', 'N', '!', ',', 'o', 'W', '+', '#', '0', 'y', 'd', 'j', 'h', 'I', '-', 'Z', 'F', 'Y', 'g', 'P', 'V', 'e', 'z', 'R', '5', 'p', '1', '?', 'x', 'c', 'D', '.', 'C', 'K', 'w', 'v', 'G', '(', 'T', '3', 'S', ':', 'i', 'A', 'E', 'a', 'U', '6', 'b', 't', '*', '9', '8', 'l', '4', ';', 'X', '2', 's', '7', 'r', 'J', 'O', 'Q', 'B', 'n', 'f', '&', 'u', '"', 'L', 'H'}
char_list = sorted(list(char_set))


# In[ ]:


def process_image(img):
    """
    Converts image to shape (32, 128, 1) & normalize
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    w, h = img.shape

    # Aspect Ratio Calculation
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape

    img = img.astype('float32')

    # Converts each to (32, 128, 1)
    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape

    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape

    if h > 128 or w > 32:
        dim = (128,32)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)
    img = np.expand_dims(img, axis=2)

    # Normalize
    img = img / 255
    return img


# In[ ]:


st.title("hand written text prediction")


# In[ ]:


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


# In[ ]:


from PIL import Image


# In[ ]:


def load_image_into_cv2(uploaded_file):
    # Convert the uploaded file to an OpenCV image
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Image not loaded. Check the path: {image_path}")

    image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    return image_cv


# In[ ]:


def predictText(predictable_img):
    predictable_array_images=[]
    predictable_img2 = process_image(predictable_img)
    predictable_array_images.append(predictable_img2)
    print(predictable_array_images[0].shape)
    bimbon_images = np.array(predictable_array_images)
    bimbon_images = tf.expand_dims(bimbon_images, axis=0)
    print(bimbon_images[0].shape)
    # predict outputs on validation images
    prediction = mj.predict(bimbon_images[0])

    print(prediction)

    # use CTC decoder
    decoded = K.ctc_decode(prediction,   
                           input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                           greedy=True)[0][0]

    out = K.get_value(decoded)
    print('out is ',out)

    for i, x in enumerate(out):
        print("predicted text = ", end = '')
        S=""
        for p in x:
            if int(p) != -1:
                #print(char_list[int(p)], end = '')
                S+=char_list[int(p)]
        new_shape = (32, 128)  # Example new shape
        reshaped_tensor = tf.reshape(bimbon_images[0], new_shape)
        plt.imshow(reshaped_tensor,cmap=plt.cm.gray)
        plt.show()
        return S
        print('\n')


# In[ ]:


import re

def starts_with_non_alpha(s):
    # Check if the string starts with any non-alphabet character
    return bool(re.match(r'^[^a-zA-Z]', s))


# In[ ]:


def HWContours(cnts):
    # Load the image into OpenCV format
    image_cv = load_image_into_cv2(uploaded_file)
    # Eliminate horizontal lines and get the masked output
    image = eliminate_horizontal_lines(image_cv)
    
    height, width = image.shape[:2]
    
    no_of_c = 0
    list_strings = []
    
    # Define boundary limits for the central 80% area
    top_limit = int(0.1 * height)
    bottom_limit = int(0.8 * height)
    left_limit = int(0.06 * width)
    right_limit = int(0.9 * width)
    
    print('Number of contours:', len(cnts))
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h > 1 and w > 1:
            if (top_limit < y < bottom_limit) and (left_limit < x < right_limit):
                roi=image[y:y+h,x:x+w]
                predictable_img = cv2.resize(roi, (300,128))

                # Resize the image to the required size (64, 64)
                test_image = cv2.resize(predictable_img, (64, 64))
                test_image=np.expand_dims(test_image,axis=0)
                plt.imshow(predictable_img,cmap='gray')

                result=cnn.predict(test_image)

                if result[0][0]==1:
                      prediction='HW'
                      no_of_c+=1
                else:
                      prediction='DT'

                if prediction == 'HW':


                    S=""
                    
                    predictable_img = cv2.cvtColor(predictable_img, cv2.COLOR_BGR2GRAY)
                    S=predictText(predictable_img)
                    if len(S)>2 and starts_with_non_alpha(S)==False:
                        
                        boxes=cv2.rectangle(image, (x, y), (x+w, y+h), (36, 255, 12), 2)
                        cv2.rectangle(image, (x, y), (x+w, y+h), (36, 255, 12), 2)
                        list_strings.append(S)


    print('no of handwritten contors are ',no_of_c)
    st.image(boxes, caption='Bounding boxes image (OpenCV format)', use_column_width=True, channels='BGR')
    return list_strings


# In[ ]:


def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Thresholding to create a binary image
    _, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Create a kernel for the morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 2))
    
    # Dilate the image
    dilate = cv2.dilate(binary, kernel, iterations=2)    
    
    return dilate


# In[ ]:


def eliminate_horizontal_lines(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Thresholding to create a binary image
    _, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw on
    output_image = image.copy()

    # Define the yellow color (BGR format)
    yellow_color = (217, 234, 238)  # OpenCV uses BGR format

    # Iterate through contours and highlight long horizontal lines
    for contour in contours:
        length = cv2.arcLength(contour, True)  # Calculate contour length
        if length < 100:  # Length threshold for filtering long contours
            # Draw the contour in yellow color
            cv2.drawContours(output_image, [contour], -1, yellow_color, thickness=cv2.FILLED)

    return output_image  # Return the modified image


# In[ ]:


if uploaded_file is not None:
    
    # Load the image into OpenCV format
    image_cv = load_image_into_cv2(uploaded_file)
    
    
    # Eliminate horizontal lines and get the masked output
    output_image = eliminate_horizontal_lines(image_cv)

    # Display the original and processed images using Streamlit
    st.image(image_cv, caption='Uploaded Image', use_column_width=True, channels='BGR')
    st.image(output_image, caption='Processed Image (Horizontal Lines Removed)', use_column_width=True, channels='BGR')
    
    
    #grayscaling the image
    gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    
    #blurring the image
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    
    #thresholding the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    #kernel of image
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 2))
    
    #dilate the image
    dilate = cv2.dilate(thresh, kernal, iterations=2)
    
    
    
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])

    
    
    list_strings_1=HWContours(cnts)
    print(list_strings_1)
    
    
    for l in range(len(list_strings_1)):
        prediction=list_strings_1[l]
        st.write(f"Prediction: {prediction}")
  


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# #  
#     # Define the border to be removed (example values)
#     top_border = 0
#     bottom_border = 0
#     left_border = 0
#     right_border = 0
#     
#     # Crop the image using numpy slicing
#     cropped_img = image_cv[top_border:image_cv.shape[0] - bottom_border, left_border:image_cv.shape[1] - right_border]
#     
#     # Convert the original image to grayscale for contour detection
#     gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
#     
#     #finding the contours
#     cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#     cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
# 
#     
#     
#     list_strings_1=HWContours(valid_contours)
#     print(list_strings_1)
#     
#     
#     for l in range(len(list_strings_1)):
#         prediction=list_strings_1[l]
#         st.write(f"Prediction: {prediction}")
#         

# # Define the height limits for top and bottom 10%
#     top_limit = int(height * 0.1)
#     bottom_limit = int(height * 0.9)
#     
#     # Define the width limits for left and right 20%
#     left_limit = int(width * 0.1)
#     right_limit = int(width * 0.8)
# 
#     # List to hold valid contours
#     valid_contours = []
# 
#     for contour in cnts:
#         # Get the bounding box of the contour
#         x, y, w, h = cv2.boundingRect(contour)
# 
#     # Check if the contour is within the top or bottom 10% of the image height
#         if y < top_limit or (y + h) > bottom_limit:
#                 print("Skipping contour at position:", y)
#                 continue  # Skip further processing for this contour
# 
#             # Check if the contour is within the left or right 20% of the image width
#         if x < left_limit or (x + w) > right_limit:
#                 print("Skipping contour at position:", x)
#                 continue  # Skip further processing for this contour
# 
#         # If valid, append to the list
#         valid_contours.append(contour)
#     

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





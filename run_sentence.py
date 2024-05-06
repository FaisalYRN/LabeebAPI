import torch
from PIL import  ImageFont, ImageDraw, Image
from torchvision.transforms import functional as F
from models.experimental import attempt_load
from utils.general import non_max_suppression
import cv2
import numpy as np
# from collections import Counter
import time

import arabic_reshaper
from bidi.algorithm import get_display



# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained YOLOv5 model on the selected device
model = attempt_load('sign_yolov5s.pt', device)  # Ensure model is in inference mode

# mapping dictionary with arabic alphabets
names = {
    0: 'ع',    # Ain
    1: 'ء',    # Alif
    2: 'ا',    # Alef
    3: 'ب',    # Ba
    4: 'د',    # Dal
    5: 'ذ',    # Dha
    6: 'ض',    # Dhad
    7: 'ف',    # Fa
    8: 'غ',    # Gaaf
    9: 'غ',    # Ghain
    11: 'ح',    # Ha
    10: 'ه',   # Haa
    12: 'ج',   # Jeem
    13: 'ك',   # Kaaf
    14: 'خ',   # Khaa
    15: 'ل',   # La
    16: 'ل',   # Laam
    17: 'م',   # Meem
    18: 'ن',   # Nun
    19: 'ر',   # Ra
    20: 'ص',   # Saad
    21: 'س',   # Seen
    22: 'ش',   # Sheen
    23: 'ت',   # Ta
    24: 'ت',   # Taa
    25: 'ث',   # Thaa
    26: 'ذ',   # Thal
    27: 'ط',   # Toot
    28: 'و',   # Waw
    29: 'ي',   # Ya
    30: 'ي',   # Yaa
    31: 'ز'    # Zay
}


# Start the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or change to the desired webcam index if you have multiple cameras

d = []
start_time = time.time()
word_list = []
key_pressed = False
final_word_list = []




while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    image_with_text = np.array(img)

    
    # Preprocess the frame and move it to the selected device
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

    # font file for Arabic language
    arabic_font = ImageFont.truetype('Amiri-Regular.ttf', size=60)

    # Perform inference
    results = model(img_tensor)[0]
    
    # Apply non-maximum suppression to get rid of redundant detections
    results = non_max_suppression(results, conf_thres=0.35, iou_thres=0.5)
    
    # Extract detected alphabets and their positions

    results = results[0].tolist()
    freq  = 0
    result = []


    cv2.putText(image_with_text, "PRESS 'w' TO VIEW WORD" ,(320, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(image_with_text, "PRESS 'r' TO reset" ,(320, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)


    # print(results)
    # for result in results:
    for obj in results:
        label = obj[-1]
        # print(int(label.tolist()))

        num = int(label)

        alpha = f"{names[num]}"
        
        x1, y1, x2, y2 = obj[:4]
    
        draw.text((50, 50), alpha, font=arabic_font, fill='white') #shows alphabets
        image_with_text = np.array(img) #array to create image of text

        cv2.rectangle(image_with_text, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

        d.append(alpha)
        print(d)

        # calculates frequency of alphabet detected and keeps the letter with specified frequency
        for i, elem in enumerate(d):
            # print(i,elem)
            freq += 1
            if freq == 10:
                result.append(elem)
            if elem != d[i-1]:
                freq = 0 

        # print(result)

        word = "".join(result)

        word = f"{word}"

        #arabic reshaper amd display for proper display of arabic words
        reshaped_p = arabic_reshaper.reshape(word)
        word = get_display(reshaped_p)

        # print(word)
        # w.append(word)

        if word not in word_list:
            word_list.append(word)

            print("WORD LIST: ", word_list)
            w = word_list[-1]

      
    key = cv2.waitKey(1) & 0xFF  # Mask to get the least significant 8 bits

    if len(results) == 0 and key == ord('s'):
        # press s to view word and add another word to form a sentence
        
        if word_list[-1] not in final_word_list:
            # reshaped_p = arabic_reshaper.reshape(word_list[-1])
            # word_list[-1] = get_display(reshaped_p)
            final_word_list.append(word_list[-1])
            print(" FINAL WORD LIST: ", final_word_list)
            if len(final_word_list) > 1:
                final_word_list.reverse()
            sentence = ' '.join(final_word_list)
        
            draw.text((200, 200), sentence, font=arabic_font, fill='white')

            image_with_text = np.array(img)
            key_pressed = True
            # word = ''
            d = []
            # print(elapsed_time)
        else:
            sentence = ' '.join(final_word_list)
        
            draw.text((200, 200), sentence, font=arabic_font, fill='white')

            image_with_text = np.array(img)
            key_pressed = True
            # word = ''
            d = []    

    
    
    if len(results) == 0 and key == ord('w'):
        #press w to view word
    
        word_list.append(word)
        draw.text((200, 200), word, font=arabic_font, fill='white')

        image_with_text = np.array(img)
        key_pressed = True

    if key_pressed and key == ord('r'):
        word = ''
        d = []

        

 
    # Display the frame with detections

    cv2.imshow('Webcam', image_with_text)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
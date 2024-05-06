from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F
from models.experimental import attempt_load
from utils.general import non_max_suppression
import torch
import arabic_reshaper
from bidi.algorithm import get_display
import os


app = Flask(__name__)

# Initialize global variables
cap = None
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
word_list = []
final_word_list = []
d = []
key_pressed = False

# Load the YOLOv5 model
model = attempt_load('sign_yolov5s.pt', device)

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
cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index_latest.html')

@app.route('/keypress', methods=['POST'])
def handle_key_press():
    global key_pressed, word_list, final_word_list

    print("KEY PRESSED: ", word_list)

    key = request.form['key']
    print("KEY: ", key)
    # print("REQUEST:" ,request.form['word'])
    if key == 'w':
        
        # word_list.append(request.form['word'])
        current_word = request.form['word']  # Update the current word
        # print(current_word)
        # word_list.append(current_word)

        # print(word_list[-1])
        # current_word = word_list[-1]
        # print("CURNT : ", current_word)
        key_pressed = True
        # return current_word

    elif key == 's':
        current_sentence = request.form['sentence']
        key_pressed = True


    elif key == 'r':
        word_list = []
        final_word_list = []
        key_pressed = False

    return jsonify({'status': 'success'})

def process_frame():
    global cap, model, device, word_list, final_word_list, key_pressed, d

    ret, frame = cap.read()
    if not ret:
        return None

    frame = cv2.resize(frame, (640, 480))
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    image_with_text = np.array(img)

    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
    arabic_font = ImageFont.truetype('Amiri-Regular.ttf', size=60)
    results = model(img_tensor)[0]
    results = non_max_suppression(results, conf_thres=0.35, iou_thres=0.5)
    results = results[0].tolist()
    result = []
    freq  = 0



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

        print(result)

        word = "".join(result)

        word = f"{word}"

        #arabic reshaper amd display for proper display of arabic words
        reshaped_p = arabic_reshaper.reshape(word)
        word = get_display(reshaped_p)

        print(word)
        # w.append(word)

        if word not in word_list:
            word_list.append(word)

            print("WORD LIST: ", word_list)
            w = word_list[-1]


    if key_pressed:
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

    # if len(results) == 0 and key == ord('v'):
    if key_pressed:
        # word_list.append(word)
        draw.text((200, 200), word_list[-1], font=arabic_font, fill='white')

        image_with_text = np.array(img)
        key_pressed = False

        

    return image_with_text

@app.route('/video_feed')
def video_feed():
   
    def generate():
        while True:
            frame = process_frame()
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    return app.response_class(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT environment variable if available, else default to 5000
    app.run(debug=True, host='0.0.0.0', port=port)

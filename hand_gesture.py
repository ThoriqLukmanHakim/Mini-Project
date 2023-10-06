import cv2
import mediapipe as mp
import pandas as pd  
import numpy as np
from flask import Flask, render_template, Response

hand = Flask(__name__)

def image_processed(hand_img):

    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Flip the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    # accessing MediaPipe solutions
    mp_hands = mp.solutions.hands

    # Initialize Hands
    hands = mp_hands.Hands(static_image_mode=True,
    max_num_hands=1, min_detection_confidence=0.7)

    # Results
    output = hands.process(img_flip)

    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        #print(data)
        data = str(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)
                        
        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return(clean)
    except:
        return(np.zeros([1,63], dtype=int)[0])

def generateframes():
    import pickle
    # load model
    with open('model.pkl', 'rb') as f:
        svm = pickle.load(f)


    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    i = 0    
    while True:
        
        success, frame = cap.read()

        if not success:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # frame = cv.flip(frame,1)
        else:
            data = image_processed(frame)
            
            # print(data.shape)
            data = np.array(data)
            y_pred = svm.predict(data.reshape(-1,63))
            print(y_pred)
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # org
            org = (50, 100)
            
            # fontScale
            fontScale = 3
            
            # Blue color in BGR
            color = (255, 0, 0)
            
            # Line thickness of 2 px
            thickness = 5
            
            # Using cv2.putText() method
            frame = cv2.putText(frame, str(y_pred[0]), org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)
            # cv2.imshow('frame', frame)
            # if cv2.waitKey(1) == ord('q'):
            #     break
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else :
            break
    cap.release()

@hand.route('/')
def index():
    return render_template('index.html')

@hand.route('/video_feed')
def video_feed():
    return Response(generateframes(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    hand.run(debug=True)

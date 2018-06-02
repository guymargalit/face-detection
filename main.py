import numpy as np
import cv2
from threading import Lock
from threading import Lock
from flask import Flask, render_template, session, request
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()

def background_thread():
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
        )  
        numFaces = 0    
        for (x,y,w,h) in faces:
            numFaces += 1
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) 
     
        ##############################################################
        # Display Output
        # Shows the output of camera
        cv2.rectangle(frame, ((300,frame.shape[0] -50)),(800, frame.shape[0]), (255,255,255), -1)
        if(numFaces == 1):
            text = str(numFaces) + " person is looking at me!"
        else:
            text = str(numFaces) + " people are looking at me!"    
        cv2.putText(frame, text, (300,frame.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 1,  (0,0,0), 1)    
        frame = cv2.resize(frame, (960, 540))
        socketio.emit('my_response',{'data': 'Server generated event', 'text': text},namespace='/test')

    cap.release()
    cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)

@socketio.on('my_event', namespace='/test')
def test_message(message):
    session['receive_count'] = session.get('receive_count', 0) + 1

@socketio.on('connect', namespace='/test')
def test_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(target=background_thread)


if __name__ == '__main__':
    socketio.run(app, host='10.0.0.223', debug=True)
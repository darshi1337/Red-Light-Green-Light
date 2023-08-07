import cv2
import numpy as np
import time
import pygame
import mediapipe as mp
from flask import Flask, render_template, Response

app = Flask(__name__)
pygame.init()
pygame.mixer.init()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

im1 = cv2.imread('im1.png')
im2 = cv2.imread('im2.png')

def calc_sum(landmarkList):
    tsum = 0
    for i in range(11, 33):
        tsum += (landmarkList[i].x * 480)
    return tsum

def calc_dist(landmarkList):
    return (landmarkList[28].y * 640 - landmarkList[24].y * 640)

def is_visible(landmarkList, visibility_threshold=0.5):
    if landmarkList[28].visibility > visibility_threshold and landmarkList[24].visibility > visibility_threshold:
        return True
    return False

def show_message(frame, message, color, progress=None):
    frame_height, frame_width, _ = frame.shape
    textsize = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
    text_x = int((frame_width - textsize[0]) / 2)
    text_y = int((frame_height + textsize[1]) / 2)
    cv2.putText(frame, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

    if progress is not None:
        cv2.putText(frame, "Progress: {}%".format(progress), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Main Window", frame)

def play_sound(sound_file):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def process_pose():
    cPos = 0
    startT = 0
    endT = 0
    thresh = 180
    isInit = False
    cStart, cEnd = 0, 0
    isCinit = False
    inFrame = 0
    inFrameCheck = False
    isDead = False
    death_count = 0
    max_death_count = 10

    while death_count < max_death_count:
        _, frm = cap.read()
        frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        frm = cv2.cvtColor(frm, cv2.COLOR_RGB2BGR)
        cv2.putText(frm, "Press 'ESC' to quit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if not inFrameCheck:
            try:
                if is_visible(res.pose_landmarks.landmark):
                    inFrame = 1
                    inFrameCheck = True
                else:
                    inFrame = 0
            except:
                print("You are not visible at all")

        if inFrame == 1:
            if not isInit:
                play_sound('greenLight.wav')
                currWindow = im1
                startT = time.time()
                endT = startT
                dur = np.random.randint(1, 5)
                isInit = True

            if (endT - startT) <= dur:
                try:
                    m = calc_dist(res.pose_landmarks.landmark)
                    if m < thresh:
                        cPos += 1
                    progress = int(cPos / 100 * 100)
                    print("Current Progress is: ", progress)
                except:
                    print("Not visible")

                endT = time.time()

            else:
                if cPos >= 100:
                    print("WINNER")
                    break
                else:
                    if not isCinit:
                        isCinit = True
                        cStart = time.time()
                        cEnd = cStart
                        currWindow = im2
                        play_sound('redLight.wav')
                        userSum = calc_sum(res.pose_landmarks.landmark)

                    if (cEnd - cStart) <= 3:
                        tempSum = calc_sum(res.pose_landmarks.landmark)
                        cEnd = time.time()
                        if abs(tempSum - userSum) > 150:
                            isDead = True
                            print("DEAD ", abs(tempSum - userSum))

                    else:
                        isInit = False
                        isCinit = False

            cv2.circle(currWindow, ((55 + 6 * cPos), 280), 15, (0, 0, 255), -1)

            mainWin = np.concatenate((cv2.resize(frm, (800, 400)), currWindow), axis=0)
            show_message(mainWin, "Red Light Green Light", (255, 255, 255), progress=progress)

        else:
            show_message(frm, "Please Make sure you are fully in frame", (0, 255, 0))

        if isDead:
            death_count += 1
            show_message(frm, "DEAD", (0, 0, 255))
            print("You died {}/{} times".format(death_count, max_death_count))
            isDead = False

        key = cv2.waitKey(1)
        if key == 27:
            print("Game ended.")
            break

    if death_count >= max_death_count:
        print("You died too many times. Game over.")

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    while True:
        _, frm = cap.read()
        frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        frm = cv2.blur(frm, (5, 5))
        mp_drawing.draw_landmarks(frm, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        ret, buffer = cv2.imencode('.jpg', frm)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/restart_game')
def restart_game():
    process_pose()
    return {'status': 'success'}

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

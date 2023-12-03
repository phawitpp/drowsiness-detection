import PySimpleGUI as sg
import time
import cv2
import math
import threading
from ultralytics import YOLO
from playsound import playsound
import numpy as np
from datetime import datetime
from opencvmultiplot import Plotter

# model setting
model = YOLO("drowsiness.pt")
class_list = model.names

# opencv setting
video = cv2.VideoCapture(0)  # Change webcam number here naja
video.set(3, 640)
video.set(4, 480)

# state
n, m, y = 0, 0, 0
curr_state = ""
detect = ""

# voice
sound_file = "wakeup.mp3"


def detect_video():
    p = Plotter(215, 200, 3)
    # save video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Change codec as needed
    output_video = cv2.VideoWriter(
        f"output_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4",
        fourcc,
        30.0,
        (int(video.get(3)), int(video.get(4))),
    )
    while True:
        global n, m, y, n_prev_list, m_prev_list, y_prev_list
        ret, frame = video.read()
        results = model(frame, stream=True)
        # Progress bar
        progress_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
        progress_bar_width = 200
        progress_bar_height = 20
        total_count = n + m + y
        n_progress = 0
        m_progress = 0
        y_progress = 0
        if total_count > 0:
            n_progress = int((n / total_count) * progress_bar_width)
            m_progress = int((m / total_count) * progress_bar_width)
            y_progress = int((y / total_count) * progress_bar_width)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                )  # convert to int values
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                match class_list[cls]:
                    case "neutral":
                        n = n + 1
                    case "microsleep":
                        m = m + 1
                    case "yawning":
                        y = y + 1
                p.multiplot([y, n, m])
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                # Detection
                cv2.putText(
                    frame,
                    class_list[cls],
                    org,
                    font,
                    fontScale,
                    color,
                    thickness,
                )

                # cal progress length x
                progress_x = frame.shape[1] - progress_bar_width - 10
                progress_y = 10
                # draw neutral
                cv2.rectangle(
                    frame,
                    (progress_x, progress_y),
                    (progress_x + n_progress, progress_y + progress_bar_height),
                    progress_colors[0],
                    -1,
                )
                # draw microsleep
                cv2.rectangle(
                    frame,
                    (progress_x, progress_y + 30),
                    (progress_x + m_progress, progress_y + progress_bar_height + 30),
                    progress_colors[1],
                    -1,
                )
                # draw yawning
                cv2.rectangle(
                    frame,
                    (progress_x, progress_y + 60),
                    (progress_x + y_progress, progress_y + progress_bar_height + 60),
                    progress_colors[2],
                    -1,
                )

                # Update status text
                status_text = f"Neutral: {math.floor(n)}"
                status_text_org = (
                    10,
                    30,
                )
                status_text2 = f"Microsleep: {math.floor(m)}"
                status_text_org2 = (
                    150,
                    30,
                )
                status_text3 = f"Yawning: {math.floor(y)}"
                status_text_org3 = (
                    290,
                    30,
                )

                # State parameters
                cv2.putText(
                    frame,
                    status_text,
                    status_text_org,
                    font,
                    0.5,
                    (0, 255, 0),
                    thickness,
                )
                cv2.putText(
                    frame,
                    status_text2,
                    status_text_org2,
                    font,
                    0.5,
                    (0, 0, 255),
                    thickness,
                )
                cv2.putText(
                    frame,
                    status_text3,
                    status_text_org3,
                    font,
                    0.5,
                    (255, 0, 0),
                    thickness,
                )
                # Predict status
                cv2.putText(
                    frame,
                    "Predict: " + detect,
                    (10, 80),
                    font,
                    1,
                    color,
                    thickness,
                )
        cv2.imshow("Webcam", frame)
        output_video.write(frame)

        if cv2.waitKey(1) == ord("q"):
            break
    video.release()
    output_video.release()
    cv2.destroyAllWindows()


def counting_state():
    while True:
        global random_state
        global n, m, y, n_prev_list, m_prev_list, y_prev_list
        global detect
        print("state n: ", n, " m:", m, " y: ", y)
        detect = (
            "neutral"
            if n >= (n + m + y) * 0.8
            else (
                "microsleep"
                if m >= (n + m + y) * 0.35
                else ("yawning" if y >= (n + m + y) * 0.35 else "neutral")
            )
        )
        print("Detect!", detect)
        if detect == "microsleep" or detect == "yawning":
            print("Wake up!")
            playsound(sound_file)
        n, m, y = 0, 0, 0
        time.sleep(2)


# Set PySimpleGUI theme for a more appealing appearance
sg.theme("LightBlue3")

# Define the layout for the PySimpleGUI window with modified styles
layout = [
    [sg.Text("Drowsiness", font=("Helvetica", 20), text_color="black")],
    [
        sg.Button("Load Video", size=(10, 2), font=("Helvetica", 12)),
        sg.Button("Use Webcam", size=(10, 2), font=("Helvetica", 12)),
    ],
]

# Create the PySimpleGUI window
window = sg.Window("Drowsiness Detection", layout, resizable=True, finalize=True)


# # Function to update the Image element with the current frame
# def update_frame():
#     while True:
#         event, values = window.read(
#             timeout=20
#         )  # Check for events every 20 milliseconds
#         if event == sg.WIN_CLOSED:
#             break
#         ret, frame = video.read()
#         if ret:
#             imgbytes = cv2.imencode(".png", frame)[1].tobytes()
#             window["-IMAGE-"].update(data=imgbytes)


# Event loop for button clicks
while True:
    event, _ = window.read()
    if event == sg.WIN_CLOSED:
        break
    elif event in ("Load Video", "Use Webcam"):
        # Update the video capture source based on the chosen event (Load Video or Use Webcam)
        if event == "Load Video":
            video_path = sg.popup_get_file(
                "Select a video file", file_types=(("Video Files", "*.mp4;*.avi"),)
            )
            if video_path:
                video = cv2.VideoCapture(video_path)
        elif event == "Use Webcam":
            video = cv2.VideoCapture(0)  # Change webcam number if necessary

        # # Start the thread to continuously update the video feed
        # thread = threading.Thread(target=update_frame)
        # thread.start()

        # Start the threads for video processing and state counting
        thread_1 = threading.Thread(target=detect_video)
        thread_2 = threading.Thread(target=counting_state)

        thread_1.start()
        thread_2.start()

        thread_1.join()
        thread_2.join()

window.close()

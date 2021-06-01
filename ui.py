import tkinter as tk
from tkinter import filedialog, messagebox, Button, Label, LEFT

import PIL.Image
import PIL.ImageTk
import cv2
import numpy as np

from predict import get_keypoints, draw, get_fall_prediction


def upload_video():
    global vid

    video_path = filedialog.askopenfilename(title='Select video to process.',
                                            filetypes=[('Video Files', ['*.mp4', '*.mov', '*.avi'])])
    vid = cv2.VideoCapture(video_path)

    if not vid:
        messagebox.showinfo('ERROR', 'ERROR!\nSomething went wrong, please restart.')
        return

    build_window()
    return


def use_web():
    global vid
    vid = cv2.VideoCapture(0)
    if not vid:
        messagebox.showinfo('ERROR', 'ERROR!\nSomething went wrong, please restart.')
        return
    build_window()
    return


def build_window():
    global window_height, window_width, vid, root, top_left_fr, \
        top_right_fr, video_fr, upload_bttn, cam_bttn, status
    window_width = 1280
    window_height = int(720 * 1.25)
    if vid:
        window_width = max(int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), window_width)
        window_height = max(int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT) * 1.25), window_height)

    root.geometry(str(window_width) + 'x' + str(window_height))

    top_left_fr = tk.Frame(root, bg="#474747")
    top_left_fr.place(relx=0.01, rely=0.02, relwidth=0.1, relheight=0.2)

    top_right_fr = tk.Frame(root, bg="#474747")
    top_right_fr.place(relx=0.12, rely=0.02, relwidth=0.85, relheight=0.2)

    video_fr = tk.Frame(root, bg="#474747")
    video_fr.place(relx=0.01, rely=0.25, relwidth=0.96, relheight=0.7)

    upload_bttn = Button(top_left_fr, text="UPLOAD", bg="#D2E59E", fg="black", font=("Arial BOLD", 15),
                         command=upload_video)
    upload_bttn.place(relx=0.1, rely=0.07, relwidth=0.8, relheight=0.4)

    cam_bttn = Button(top_left_fr, text="CAM FEED", bg="#D2E59E", fg="black", font=("Arial BOLD", 15),
                      command=use_web)
    cam_bttn.place(relx=0.1, rely=0.55, relwidth=0.8, relheight=0.4)

    status = tk.Label(top_right_fr, bg="#E0E0CE", fg="#AD343E", font=("Arial BOLD", 15), justify=LEFT)
    status.place(relx=0.05, rely=0.1, relwidth=0.9, relheight=0.8)
    if vid:
        main_processing()
    return


def update_list():
    global frame_list, curr_frame
    frame_list = np.vstack([frame_list, curr_frame])


def main_processing():
    global vid, window_height, window_width, video_flow, root, canvas, i, window, stride, falls
    width = int(window_width * 0.57)
    height = int(window_height * 0.86)
    dim = (width, height)
    status_str = f'Falls overall: {falls}'

    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("No image!")
    try:
        global frame_list, curr_frame
        frame, pose_kps = get_keypoints(frame)
        frame = draw(frame, pose_kps)
        curr_frame = np.expand_dims(pose_kps[:, :2].flatten(), axis=0)
        update_list()
        frame_input = np.expand_dims(np.array(frame_list), axis=0)
        if frame_list.shape[0] == 50:
            res = get_fall_prediction(frame_input)
            if res[0][1] > 0.6:
                falls += 1
                status_str = f'FALL DETECTED\n\nOverall: {falls}'
            frame_list = frame_list[stride:]
    except Exception as e:
        print(e)

    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    processed_frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))

    video_flow = Label(video_fr, bg="#474747")
    video_flow.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.9)
    video_flow.image = processed_frame
    video_flow.configure(image=processed_frame)
    status["text"] = status_str
    root.after(5, main_processing)


if __name__ == "__main__":
    global top_left_fr, top_right_fr, video_fr, cam_bttn, upload_bttn, status, video_flow, window_height, window_width,\
        vid, window, stride, frame_list, falls, curr_frame

    top_left_fr = top_right_fr = video_fr = cam_bttn = upload_bttn \
        = status = video_flow = window_height = window_width = vid = None
    window = 50
    stride = 20
    falls = 0
    frame_list = np.empty((1, 34))

    root = tk.Tk()
    root.configure(bg="#E0E0CE")
    root.title("Fall Detection")

    build_window()

    root.mainloop()

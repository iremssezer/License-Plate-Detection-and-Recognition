import tkinter as tk
from tkinter import filedialog
import sqlite3 as sql
import cv2
import PIL.Image, PIL.ImageTk
import tkinter.ttk as ttk
from ultralytics import YOLO
import pandas as pd
import re
from PIL import Image
import numpy as np
import os
import threading
import pytesseract

if os.path.exists("plates.sqlite"):
        os.remove("plates.sqlite")


def detect_objects(frame):
    results = model.predict(frame)
    return results


def process_video(selected_video):
    cap = cv2.VideoCapture(selected_video)

    db = sql.connect("plates.sqlite")
    cursor = db.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS plates (id, plate)")

    id_counter = 1
    plate_counter = {}
    plate_counter_list, last_list, keys = [], [], []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        results = detect_objects(frame)
        draw = model(frame)
        annoted_frame = draw[0].plot()
        pil_image = Image.fromarray(cv2.cvtColor(annoted_frame, cv2.COLOR_BGR2RGB))

        for r in results:
            boxes = r.boxes.xyxy

            if len(boxes) > 0:
                boxes_df = pd.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2'])

                for index, row in boxes_df.iterrows():
                        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cropped = frame[y1:y2, x1:x2]


                        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                        blur = cv2.GaussianBlur(gray, (3, 3), 0)
                        thresh = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]



                        pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
                        crop_img_text = pytesseract.image_to_string(thresh, lang='eng')


                        print("text: ", crop_img_text)
                        if len(crop_img_text) > 5:
                            if not re.search(r'[!*\'+()%/?~|,«—:\[\]§.°¥a-z]', crop_img_text):
                                if crop_img_text[0] == ' ' or crop_img_text[0] == '-':
                                    crop_img_text = crop_img_text[1:]
                                if crop_img_text[-1] == ' ' or crop_img_text[-1] == '-':
                                    crop_img_text = crop_img_text[:-1]
                                plate_counter_list.append(crop_img_text)
                                if crop_img_text in plate_counter:
                                    plate_counter[crop_img_text] += 1
                                else:
                                    plate_counter[crop_img_text] = 1
                                for key, value in plate_counter.items():
                                    if value > 3:
                                        last_list.append(key)
                                        keys.append(key)
                                        if len(last_list) > 1 and last_list[-1] != last_list[-2]:
                                            last_list = [last_list[-1]]
                                        cursor.execute("insert into plates (id, plate) values (?,?)", (id_counter, last_list[-1]))
                                        id_counter = id_counter + 1
                                        db.commit()
                                for key in keys:
                                    if key in plate_counter:
                                        del plate_counter[key]



            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    db.close()
    #return pil_image
model = YOLO('best.pt')


root = tk.Tk()
root.title("Video Oynatıcı")

selected_video = ""

video_player_frame = tk.Frame(root)
video_player_frame.pack()

def create_video_player():
    global video_player_frame
    video_player_frame.destroy()

    video_player_frame = tk.Frame(root)
    video_player_frame.pack()

    cap = cv2.VideoCapture(selected_video)

    fps = cap.get(cv2.CAP_PROP_FPS)

    video_player = tk.Label(video_player_frame)
    video_player.pack()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame)
        img_tk = PIL.ImageTk.PhotoImage(image=img)
        video_player.configure(image=img_tk)
        video_player.image = img_tk

        root.update()

        delay = int(1000 / fps)
        cv2.waitKey(delay)

    cap.release()

def select_video():
    global selected_video, video_player_frame
    filename = filedialog.askopenfilename(filetypes=[("Video dosyaları", "*.mp4;*.avi")])
    selected_video = filename
    create_video_player()

button = tk.Button(root, text="Video Seç", command=select_video)
button.pack()

data_table = ttk.Treeview(root)
data_table["columns"] = ("ID", "Plate No")

data_table.heading("#0", text="")
data_table.heading("ID", text="ID")
data_table.heading("Plate No", text="Plaka No")
#data_table.heading("Time", text="Saat")

data_table.column("#0", width=1)
data_table.column("ID", width=100)
data_table.column("Plate No", width=100)
#data_table.column("Time", width=100)

def show_data():
    db = sql.connect("plates.sqlite")
    cursor = db.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS plates (id, plate)")
    data_table.delete(*data_table.get_children())
    cursor.execute("SELECT * FROM plates")
    data = cursor.fetchall()
    for row in data:
        data_table.insert("", tk.END, text="", values=row)

    data_table.update()
    cursor.close()
    db.close()
    root.after(200, show_data)


data_table.pack()
show_data()
def process_video_thread():
    db = sql.connect("plates.sqlite")
    cursor = db.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS plates (id, plate)")
    t = threading.Thread(target=process_video, args=(selected_video,))
    t.start()

process_button = tk.Button(root, text="Video İşle", command=process_video_thread)
process_button.pack()

root.mainloop()


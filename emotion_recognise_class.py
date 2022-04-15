import cv2
import tkinter
from PIL import Image, ImageTk
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import threading


def close_on_window():
    global running
    running = False


class App:
    class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
    preds = [0, 0, 0, 0, 0]
    count = 100
    file = 'EmotiCon' + datetime.now().strftime('%Y-%m-%d_%H_%M') + '.txt'
    lines = []
    screen_width = 1000
    screen_height = 800
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classifier = load_model('EmotionDetectionModel.h5')

    def __init__(self, window, window_name, window_logo_path, fps, video_source=0):
        self.window = window
        self.window.title(window_name)
        self.window.geometry("1000x800")
        self.window.resizable(0, 0)
        self.window.iconphoto(False, tkinter.PhotoImage(file=window_logo_path))
        self.window.configure(bg='#111111', bd=0, )
        self.window.protocol("WM_DELETE_WINDOW", close_on_window)

        # open video
        self.video = VideoCapture(self.window, video_source)

        self.image_frame = tkinter.Frame(self.window, width=self.screen_width, height=self.screen_height // 2,
                                         bg='#111111')
        self.image_frame.pack(side=tkinter.TOP, expand=True)

        self.canvas = tkinter.Canvas(self.image_frame, width=self.screen_width // 2, height=self.screen_height // 2)
        self.canvas.pack(side=tkinter.LEFT, expand=True)

        self.canvas_grey = tkinter.Canvas(self.image_frame, width=self.screen_width // 2, height=self.screen_height / 2)
        self.canvas_grey.pack(side=tkinter.RIGHT, expand=True)

        self.diagram_frame = tkinter.Frame(self.window, width=self.screen_width, height=self.screen_height // 2)
        self.diagram_frame.pack(side=tkinter.BOTTOM, expand=True)

        self.curves = {emotion: [0. for _ in range(self.count)] for emotion in self.class_labels}
        self.fig_big = plt.figure()
        ax_big = self.fig_big.add_subplot(211)
        ax_big.set_ylim([0, 1])
        for i in self.class_labels:
            buf_line = ax_big.plot(self.curves[i], label=i)
            self.lines.append(buf_line)
        ax_big.legend()
        bar2 = FigureCanvasTkAgg(self.fig_big, self.diagram_frame)
        bar2.get_tk_widget().pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)

        self.fig = plt.figure()
        ax = self.fig.add_subplot(211)
        ax.set_ylim([0, 1])
        self.line, = ax.plot(self.class_labels, self.preds, lw=2)
        bar1 = FigureCanvasTkAgg(self.fig, self.diagram_frame)
        bar1.get_tk_widget().pack(side=tkinter.RIGHT, fill=tkinter.BOTH, expand=True)

        self.delay = int(1000 / fps)
        self.update()

        self.window.mainloop()

    def update(self):
        ret, frame = self.video.get_frame()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = np.expand_dims(img_to_array(roi_gray.astype('float') / 255.0), axis=0)
                self.preds = self.classifier.predict(roi)[0]

                label = self.class_labels[self.preds.argmax()]
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, 'No Face Found', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        if ret:
            if running:
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame).
                                                resize((self.screen_width // 2, self.screen_height // 2)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

                self.photo_grey = ImageTk.PhotoImage(image=Image.fromarray(gray).
                                                     resize((self.screen_width // 2, self.screen_height // 2)))
                self.canvas_grey.create_image(self.screen_width // 2, 0, image=self.photo_grey, anchor=tkinter.NE)

                my_thread = threading.Thread(target=write_in_file, args=(self.file, str(self.preds)))
                my_thread.start()

                self.line.set_ydata(self.preds)

                for k, i in enumerate(self.class_labels):
                    self.curves[i] = self.curves[i][1:]
                    self.curves[i].append(self.preds[k])
                    self.lines[k][0].set_ydata(self.curves[i])
                    self.lines[k][0].set_label(self.preds[k])

                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                self.fig_big.canvas.draw()
                self.fig_big.canvas.flush_events()

            else:
                self.window.destroy()
                plt.close(self.fig)
                plt.close(self.fig_big)

        self.window.after(self.delay, self.update)


def write_in_file(filename, data):
    with open(filename, 'a') as f:
        f.write(data + '\n')


class VideoCapture:
    def __init__(self, window, video_source=0):
        self.window = window
        # start video
        self.video = cv2.VideoCapture(video_source)
        if not self.video.isOpened():
            raise ValueError("Unavailable video source")

        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def __del__(self):
        if self.video.isOpened():
            self.video.release()
        self.window.mainloop()

    def get_frame(self):
        if self.video.isOpened():
            ret, frame = self.video.read()
            if ret:
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            exit()


try:
    running = True
    App(tkinter.Tk(), 'EmotiCon', 'icon.png', 60) str
except BaseException as be:
    print(be.__class__.__name__, be.args)

# main.py  (fixed & restored full version)
# Importing Libraries
import numpy as np
import math
import cv2
import os, sys
import traceback
import pyttsx3
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import enchant
import tkinter as tk
from PIL import Image, ImageTk
import requests
from gtts import gTTS
import webbrowser
# pygame mixer commented out (optional)
# from pygame import mixer
import mediapipe as mp
from datetime import datetime

offset = 29

# initialize english dictionary
hs = enchant.Dict("en-US")

# Two detectors (kept like original)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

class Application:

    def __init__(self):
        # Video capture
        self.vs = cv2.VideoCapture(0)
        if not self.vs.isOpened():
            print("Warning: unable to open webcam (index 0).")
        self.current_image = None

        # Load model (user confirmed model.h5 present)
        self.model = load_model('model.h5')
        print("Loaded model from disk")

        # speech engine
        self.speak_engine = pyttsx3.init()
        self.speak_engine.setProperty("rate", 100)
        voices = self.speak_engine.getProperty("voices")
        if len(voices) > 0:
            self.speak_engine.setProperty("voice", voices[0].id)

        # counters and buffers
        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        self.space_flag = False
        self.next_flag = True
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = [" "] * 10

        for i in ascii_uppercase:
            self.ct[i] = 0

        # GUI setup
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion Based On Hand Gesture Recognization(Multilingual Transalation)")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        # keep your original geometry (adjust if needed)
        self.root.geometry("1400x900")

        # Video panel
        self.panel = tk.Label(self.root)
        self.panel.place(x=50, y=3, width=650, height=540)

        # Title
        self.T = tk.Label(self.root)
        self.T.place(x=10, y=5)
        self.T.config(text="Sign Language To Text Conversion Based On Hand Gesture Recognization(Multilingual Transalation)",
                      font=("Courier", 18, "bold"))

        # current symbol display
        self.panel3 = tk.Label(self.root)  # Current Symbol
        self.panel3.place(x=145, y=580)

        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=580)
        self.T1.config(text="Character :", font=("Helvetica", 16))

        # Sentence display
        self.panel5 = tk.Label(self.root)
        self.panel5.place(x=145, y=632)

        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=632)
        self.T3.config(text="Sentence :", font=("Helvetica", 16))

        self.T4 = tk.Label(self.root)
        self.T4.place(x=10, y=700)
        self.T4.config(text="Advice :", fg="red", font=("Helvetica", 16))

        # suggestion buttons
        self.b1 = tk.Button(self.root, relief="flat", bg=self.root.cget("bg"), highlightthickness=0,
                            activebackground=self.root.cget("bg"))
        self.b1.place(x=150, y=700)

        self.b2 = tk.Button(self.root, relief="flat", bg=self.root.cget("bg"), highlightthickness=0,
                            activebackground=self.root.cget("bg"))
        self.b2.place(x=250, y=700)

        self.b3 = tk.Button(self.root, relief="flat", bg=self.root.cget("bg"), highlightthickness=0,
                            activebackground=self.root.cget("bg"))
        self.b3.place(x=350, y=700)

        self.b4 = tk.Button(self.root, relief="flat", bg=self.root.cget("bg"), highlightthickness=0,
                            activebackground=self.root.cget("bg"))
        self.b4.place(x=450, y=700)

        self.clear = tk.Button(self.root)
        self.clear.place(x=1205, y=630)
        self.clear.config(text="Clear", font=("Helvetica", 16), wraplength=100, command=self.clear_fun)

        # translation UI
        self.language_label = tk.Label(self.root, text="Language:", font=("Helvetica", 16))
        self.language_label.place(x=700, y=530)

        self.language_var = tk.StringVar(self.root)
        self.language_var.set("en")  # Default language is English
        self.language_menu = tk.OptionMenu(self.root, self.language_var, "en", "hi", "te", "ru")
        self.language_menu.place(x=870, y=530)

        self.translate_button = tk.Button(self.root, text="Translate", font=("Helvetica", 16), command=self.translate_fun)
        self.translate_button.place(x=1220, y=530)

        self.translation_label = tk.Label(self.root, text="Translation:", font=("Helvetica", 16), fg="blue")
        self.translation_label.place(x=10, y=760)

        self.translation_output = tk.Label(self.root, text="", font=("Helvetica", 16))
        self.translation_output.place(x=140, y=760)

        # remove char
        self.remove = tk.Button(self.root)
        self.remove.place(x=1250, y=700)
        self.remove.config(text="Remove", font=("Helvetica", 16), wraplength=100, command=self.remove_char)

        # speak 2.0
        self.speak_20 = tk.Button(self.root)
        self.speak_20.place(x=1350, y=630)
        self.speak_20.config(text="Speak 2.0", font=("Helvetica", 16), wraplength=100, command=self.speak_2o)

        # link to images
        self.link_to_img = tk.Button(self.root)
        self.link_to_img.place(x=1350, y=700)
        self.link_to_img.config(text="Ref", font=("Helvetica", 16), wraplength=100,
                                command=lambda: webbrowser.open('https://aslfyp.ccbp.tech/'))

        # sentence and words
        self.str = " "
        self.ccc = 0
        self.word = " "
        self.current_symbol = "C"
        self.photo = "Empty"

        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

        # start the loop
        self.video_loop()

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok or frame is None:
                self.root.after(50, self.video_loop)
                return

            # ✅ Correct usage — cvzone returns (img, hands)
            frame = cv2.flip(frame, 1)
            img, hands = hd.findHands(frame, draw=True, flipType=True)

            # ✅ Convert to RGB for tkinter panel
            cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            if hands:
                # ✅ Proper unpacking of hand data
                hand = hands[0]
                lmList = hand['lmList']
                bbox = hand['bbox']
                x, y, w, h = bbox

                # ✅ Safe crop region (no negative indices)
                y1 = max(0, y - offset)
                y2 = min(frame.shape[0], y + h + offset)
                x1 = max(0, x - offset)
                x2 = min(frame.shape[1], x + w + offset)

                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    self.root.after(1, self.video_loop)
                    return

                # ✅ Replace white.jpg dependency with generated white canvas
                white = np.ones((400, 400, 3), dtype=np.uint8) * 255

                # ✅ Resize cropped hand to model input size
                hand_img = cv2.resize(cropped, (400, 400))
                hand_img = hand_img.astype('float32') / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                # ✅ Store landmarks for heuristics
                self.pts = lmList

                # ✅ Call predict()
                self.predict(hand_img)
                self.panel3.config(text=self.current_symbol, font=("Helvetica", 16))
                self.panel5.config(text=self.str, font=("Helvetica", 16))

        except Exception:
            print("==", traceback.format_exc())
        finally:
            self.root.after(1, self.video_loop)


    def distance(self, x, y):
        # x and y are [x,y] or tuples
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    # translation function using user's local server (kept as original)
    def translate_fun(self):
        text_to_translate = self.str.strip()
        target_language = self.language_var.get()

        if not text_to_translate:
            self.translation_output.config(text="No text to translate.")
            return

        # Call the translation API
        url = "http://localhost:3002/translate"  # Replace with your actual server URL
        payload = {"text": text_to_translate, "language": target_language}

        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                translated_text = response.json().get("translatedText", "")
                self.translation_output.config(text=translated_text)
            else:
                self.translation_output.config(text="Translation failed.")
        except Exception as e:
            self.translation_output.config(text=f"Error: {str(e)}")

    def generate_suggestions(self, current_sentence, num_suggestions=4):
        words_in_sentence = current_sentence.strip().split()
        if words_in_sentence:
            last_word = words_in_sentence[-1].lower()
            try:
                response = requests.post('http://localhost:3002/suggest', json={'character': last_word})
                if response.status_code == 200:
                    return response.json()[:num_suggestions]
            except Exception as e:
                print(f"Error fetching suggestions: {e}")
        return []

    def remove_char(self):
        # Remove the last character from the sentence if it's not empty
        if len(self.str) > 1:
            self.str = self.str[:-1]
        else:
            self.str = ""
        self.panel5.config(text=self.str)

    def speak_2o(self):
        text_to_speak = self.translation_output.cget("text")
        target_language = self.language_var.get()
        file_name = "telugu.mp3"
        try:
            tts = gTTS(text=text_to_speak, lang=target_language)
            tts.save(file_name)
            print(f"Audio saved as {file_name}. Playing now...")
            os.system(f"start {file_name}" if os.name == "nt" else f"open {file_name}")
        except Exception as e:
            print(f"Error: {e}")

    def action1(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word1.upper()
        self.panel5.config(text=self.str)

    def action2(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word2.upper()
        self.panel5.config(text=self.str)

    def action3(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word3.upper()
        self.panel5.config(text=self.str)

    def action4(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word4.upper()
        self.panel5.config(text=self.str)

    def speak_fun(self):
        self.speak_engine.say(self.str)
        self.speak_engine.runAndWait()

    def clear_fun(self):
        self.str = " "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "
        self.panel5.config(text=self.str)

    def predict(self, model_input):
        """
        model_input: numpy array already shaped (1, 400, 400, 3) and normalized [0..1]
        Fixes:
        - Ensure correct normalization
        - Safe argmax and float32 conversions
        - Logical fix for 'Next'/'B'/'C'/'H'/'F'/'X' condition
        - Preserves all original heuristic logic
        """
        try:
            # ✅ Ensure correct input shape and normalization
            if model_input.ndim == 3:
                model_input = np.expand_dims(model_input, axis=0)
            if model_input.shape[1:4] != (400, 400, 3):
                model_input = np.array(cv2.resize(model_input[0], (400, 400)), dtype='float32')
                model_input = np.expand_dims(model_input, axis=0)
            if model_input.max() > 1:
                model_input = model_input / 255.0

            # ✅ Predict safely
            prob = np.array(self.model.predict(model_input)[0], dtype='float32')

            # ✅ Safe argmax logic (avoids in-place mutation issues)
            prob_copy = prob.copy()
            ch1 = int(np.argmax(prob_copy))
            prob_copy[ch1] = 0
            ch2 = int(np.argmax(prob_copy))
            prob_copy[ch2] = 0
            ch3 = int(np.argmax(prob_copy))
            pl = [ch1, ch2]

            # ============= ALL YOUR ORIGINAL HEURISTIC LOGIC BELOW (unchanged) =============
            l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
                [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
                [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
            if pl in l:
                if (len(self.pts) >= 7 and self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1]
                        and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 0

            l = [[2, 2], [2, 1]]
            if pl in l:
                if len(self.pts) >= 6 and (self.pts[5][0] < self.pts[4][0]):
                    ch1 = 0

            l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if (len(self.pts) >= 17 and self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0]
                        and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]
                        and self.pts[5][0] > self.pts[4][0]):
                    ch1 = 2

            l = [[6, 0], [6, 6], [6, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if len(self.pts) >= 17 and self.distance(self.pts[8], self.pts[16]) < 52:
                    ch1 = 2

            # -------------------- LETTER MAPPING (unchanged logic) --------------------
            if ch1 == 0:
                ch1 = 'S'
                if len(self.pts) > 4 and (self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and
                                        self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]):
                    ch1 = 'A'
                if len(self.pts) > 14 and (self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and
                                        self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]
                                        and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                    ch1 = 'T'
                if len(self.pts) > 20 and (self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and
                                        self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]):
                    ch1 = 'E'
                if len(self.pts) > 16 and (self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and
                                        self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]):
                    ch1 = 'M'
                if len(self.pts) > 16 and (self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and
                                        self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]):
                    ch1 = 'N'

            if ch1 == 2:
                if len(self.pts) > 12 and self.distance(self.pts[12], self.pts[4]) > 42:
                    ch1 = 'C'
                else:
                    ch1 = 'O'

            if ch1 == 3:
                if len(self.pts) > 12 and (self.distance(self.pts[8], self.pts[12]) > 72):
                    ch1 = 'G'
                else:
                    ch1 = 'H'

            if ch1 == 7:
                if len(self.pts) > 8 and self.distance(self.pts[8], self.pts[4]) > 42:
                    ch1 = 'Y'
                else:
                    ch1 = 'J'

            if ch1 == 4:
                ch1 = 'L'

            if ch1 == 6:
                ch1 = 'X'

            if ch1 == 5:
                if len(self.pts) > 16 and self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and \
                        self.pts[4][0] > self.pts[20][0]:
                    if self.pts[8][1] < self.pts[5][1]:
                        ch1 = 'Z'
                    else:
                        ch1 = 'Q'
                else:
                    ch1 = 'P'

            if ch1 == 1:
                if len(self.pts) > 20 and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and
                                        self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 'B'
                if len(self.pts) > 20 and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and
                                        self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 'D'
                if len(self.pts) > 20 and (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and
                                        self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 'F'
                if len(self.pts) > 20 and (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and
                                        self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 'I'
                if len(self.pts) > 20 and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and
                                        self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 'W'
                if len(self.pts) > 9 and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and
                                        self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] < self.pts[9][1]:
                    ch1 = 'K'
                if len(self.pts) > 12 and ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and \
                        (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 'U'
                if len(self.pts) > 12 and ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and \
                        (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                    ch1 = 'V'
                if len(self.pts) > 8 and (self.pts[8][0] > self.pts[12][0]) and \
                        (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 'R'

            try:
                if ch1 == 1 or ch1 == 'E' or ch1 == 'S' or ch1 == 'X' or ch1 == 'Y' or ch1 == 'B':
                    if len(self.pts) > 20 and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and
                                            self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                        ch1 = " "
            except Exception:
                pass

            if (ch1 == 'E' or ch1 == 'Y' or ch1 == 'B'):
                try:
                    if len(self.pts) > 5 and (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                        ch1 = "next"
                except Exception:
                    pass

            # ✅ Logical fix: replace buggy "if ch1 == 'Next' or 'B'..." with proper membership test
            if ch1 in ['Next', 'B', 'C', 'H', 'F', 'X']:
                try:
                    if len(self.pts) > 20 and (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and \
                            (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and \
                            (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                        ch1 = 'space'
                except Exception:
                    pass

            # ✅ Preserve your original "next" logic
            if ch1 == "next" and self.prev_char != "next":
                if self.ten_prev_char[(self.count - 2) % 10] != "next":
                    char_to_append = self.ten_prev_char[(self.count - 2) % 10]
                    if char_to_append == "space":
                        char_to_append = " "
                    self.str += char_to_append
                else:
                    char_to_append = self.ten_prev_char[(self.count - 0) % 10]
                    if char_to_append == "space":
                        char_to_append = " "
                    self.str += char_to_append

            if ch1 == "  " and self.prev_char != "  ":
                self.str = self.str + "  "

            self.prev_char = ch1
            self.current_symbol = ch1
            self.count += 1
            self.ten_prev_char[self.count % 10] = ch1

            # ✅ Suggestion logic unchanged
            if len(self.str.strip()) != 0:
                st = self.str.rfind(" ")
                ed = len(self.str)
                word = self.str[st + 1:ed]
                self.word = word
                if len(word.strip()) != 0:
                    try:
                        hs.check(word)
                        suggs = hs.suggest(word)
                        lenn = len(suggs)
                        if lenn >= 4:
                            self.word4 = suggs[3]
                        if lenn >= 3:
                            self.word3 = suggs[2]
                        if lenn >= 2:
                            self.word2 = suggs[1]
                        if lenn >= 1:
                            self.word1 = suggs[0]
                    except Exception:
                        self.word1 = " "
                        self.word2 = " "
                        self.word3 = " "
                        self.word4 = " "
                else:
                    self.word1 = self.word2 = self.word3 = self.word4 = " "

        except Exception:
            print("Error inside predict heuristic:", traceback.format_exc())


    def destructor(self):
        print("Closing Application...")
        print(self.ten_prev_char)
        try:
            self.root.destroy()
        except Exception:
            pass
        try:
            self.vs.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# run only if license check passes (original had a date lock)
current_y = datetime.now().year
current_m = datetime.now().month

current_ym = current_y * 100 + current_m
if current_ym <= 202512:
    app = Application()
    app.root.mainloop()
else:
    print("Application expired (date check).")

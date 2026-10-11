import sys
import cv2
import mediapipe as mp
import pickle
import time
import pyttsx3

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


# Load ML model
model = pickle.load(open("sign_model.pkl", "rb"))

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


class SignLanguageApp(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Sign Language Translator")
        self.setGeometry(100,100,1200,700)

        # Speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate',150)

        # Word dictionary
        self.dictionary = [
        "HELLO","HELP","THANK","THANKYOU",
        "YES","NO","GOOD","HOME","HOUSE",
        "WATER","FRIEND","FOOD","WORK",
        "WELCOME","BYE"
        ]

        # Offline translation dictionary
        self.translation_dict = {

        "HELLO":("नमस्ते","ನಮಸ್ಕಾರ"),
        "HELP":("मदद","ಸಹಾಯ"),
        "THANK":("धन्यवाद","ಧನ್ಯವಾದ"),
        "THANKYOU":("धन्यवाद","ಧನ್ಯವಾದಗಳು"),
        "YES":("हाँ","ಹೌದು"),
        "NO":("नहीं","ಇಲ್ಲ"),
        "GOOD":("अच्छा","ಒಳ್ಳೆಯದು"),
        "HOME":("घर","ಮನೆ"),
        "HOUSE":("घर","ಮನೆ"),
        "WATER":("पानी","ನೀರು"),
        "FRIEND":("दोस्त","ಸ್ನೇಹಿತ"),
        "FOOD":("खाना","ಆಹಾರ"),
        "WORK":("काम","ಕೆಲಸ"),
        "WELCOME":("स्वागत","ಸ್ವಾಗತ"),
        "BYE":("अलविदा","ಬೈ")

        }

        # Style
        self.setStyleSheet("""

        QWidget{
            background:#0f0f0f;
            color:white;
            font-family:Segoe UI;
        }

        QPushButton{
            background:#00ffc3;
            padding:10px;
            border-radius:8px;
            font-weight:bold;
            color:black;
        }

        QPushButton:hover{
            background:#00d6a8;
        }

        QListWidget{
            background:#1e1e1e;
            border-radius:8px;
        }

        QFrame{
            background:#1e1e1e;
            border-radius:10px;
        }

        """)

        # Variables
        self.sentence=""
        self.last_prediction=""
        self.frame_count=0
        self.last_letter_time=0
        self.delay=2

        main_layout = QVBoxLayout()

        title = QLabel("🤟 AI SIGN LANGUAGE TRANSLATOR")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:32px;font-weight:bold;color:#00ffc3")

        team = QLabel("Developed by Team STM32")
        team.setAlignment(Qt.AlignCenter)
        team.setStyleSheet("color:gray")

        main_layout.addWidget(title)
        main_layout.addWidget(team)

        content_layout = QHBoxLayout()

        # Camera panel
        camera_frame = QFrame()
        camera_layout = QVBoxLayout()

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)

        self.camera_label.setStyleSheet("""
        border:3px solid #00ffc3;
        border-radius:10px;
        """)

        camera_layout.addWidget(self.camera_label)
        camera_frame.setLayout(camera_layout)

        content_layout.addWidget(camera_frame,3)

        # Side panel
        side_panel = QVBoxLayout()

        letter_title = QLabel("Detected Sign")

        self.letter_label = QLabel("-")
        self.letter_label.setAlignment(Qt.AlignCenter)
        self.letter_label.setStyleSheet("""
        font-size:80px;
        font-weight:bold;
        color:#00ffc3;
        """)

        sentence_title = QLabel("English Sentence")

        self.sentence_label = QLabel("")
        self.sentence_label.setWordWrap(True)
        self.sentence_label.setStyleSheet("""
        background:#1e1e1e;
        padding:15px;
        border-radius:10px;
        font-size:20px;
        """)

        # Hindi panel
        hindi_title = QLabel("Hindi Translation")

        self.hindi_label = QLabel("")
        self.hindi_label.setWordWrap(True)
        self.hindi_label.setStyleSheet("""
        background:#1e1e1e;
        padding:12px;
        border-radius:8px;
        font-size:18px;
        """)

        # Kannada panel
        kannada_title = QLabel("Kannada Translation")

        self.kannada_label = QLabel("")
        self.kannada_label.setWordWrap(True)
        self.kannada_label.setStyleSheet("""
        background:#1e1e1e;
        padding:12px;
        border-radius:8px;
        font-size:18px;
        """)

        # Suggestions
        suggestion_title = QLabel("Word Suggestions")
        self.suggestion_box = QListWidget()

        # History
        history_title = QLabel("Gesture History")
        self.history_box = QListWidget()

        # Buttons
        self.space_btn = QPushButton("➕ Add Space")
        self.clear_btn = QPushButton("🗑 Clear Sentence")
        self.speak_btn = QPushButton("🔊 Speak Sentence")

        side_panel.addWidget(letter_title)
        side_panel.addWidget(self.letter_label)

        side_panel.addWidget(sentence_title)
        side_panel.addWidget(self.sentence_label)

        side_panel.addWidget(hindi_title)
        side_panel.addWidget(self.hindi_label)

        side_panel.addWidget(kannada_title)
        side_panel.addWidget(self.kannada_label)

        side_panel.addWidget(suggestion_title)
        side_panel.addWidget(self.suggestion_box)

        side_panel.addWidget(history_title)
        side_panel.addWidget(self.history_box)

        side_panel.addWidget(self.space_btn)
        side_panel.addWidget(self.clear_btn)
        side_panel.addWidget(self.speak_btn)

        side_panel.addStretch()

        content_layout.addLayout(side_panel,1)

        main_layout.addLayout(content_layout)

        footer = QLabel("© 2026 Team STM32")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color:#555")

        main_layout.addWidget(footer)

        self.setLayout(main_layout)

        # Button actions
        self.space_btn.clicked.connect(self.add_space)
        self.clear_btn.clicked.connect(self.clear_sentence)
        self.speak_btn.clicked.connect(self.speak_sentence)
        self.suggestion_box.itemClicked.connect(self.use_suggestion)

        # Camera
        self.cap = cv2.VideoCapture(0)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)


    # Sentence functions
    def add_space(self):

        self.sentence += " "
        self.update_sentence()


    def clear_sentence(self):

        self.sentence=""
        self.history_box.clear()
        self.update_sentence()


    def update_sentence(self):

        self.sentence_label.setText(self.sentence)
        self.update_suggestions()
        self.update_translations()


    # Speech
    def speak_sentence(self):

        text = self.sentence.strip()

        if text == "":
            return

        self.engine.say(text)
        self.engine.runAndWait()


    # Offline translation
    def update_translations(self):

        text = self.sentence.strip()

        if text == "":
            self.hindi_label.setText("")
            self.kannada_label.setText("")
            return

        words = text.split()

        hindi=[]
        kannada=[]

        for word in words:

            word=word.upper()

            if word in self.translation_dict:

                h,k = self.translation_dict[word]

            else:

                h=word
                k=word

            hindi.append(h)
            kannada.append(k)

        self.hindi_label.setText(" ".join(hindi))
        self.kannada_label.setText(" ".join(kannada))


    # Suggestions
    def get_current_word(self):

        words=self.sentence.split(" ")
        return words[-1]


    def update_suggestions(self):

        current_word=self.get_current_word()

        self.suggestion_box.clear()

        if current_word=="":
            return

        for word in self.dictionary:

            if word.startswith(current_word.upper()):
                self.suggestion_box.addItem(word)


    def use_suggestion(self,item):

        word=item.text()

        words=self.sentence.split(" ")
        words[-1]=word

        self.sentence=" ".join(words)+" "

        self.update_sentence()


    # Camera frame
    def update_frame(self):

        ret,frame=self.cap.read()

        if not ret:
            return

        frame=cv2.flip(frame,1)

        imageRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        results=hands.process(imageRGB)

        prediction=""

        if results.multi_hand_landmarks:

            for hand in results.multi_hand_landmarks:

                landmarks=[]

                for lm in hand.landmark:

                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                    landmarks.append(lm.z)

                prediction=model.predict([landmarks])[0]

                if prediction==self.last_prediction:
                    self.frame_count+=1
                else:
                    self.frame_count=0

                self.last_prediction=prediction

                current_time=time.time()

                if self.frame_count==10 and (current_time-self.last_letter_time)>self.delay:

                    self.sentence+=prediction

                    self.last_letter_time=current_time
                    self.frame_count=0

                    self.update_sentence()
                    self.history_box.addItem(prediction)

                mp_draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)

        if prediction=="":
            self.letter_label.setText("-")
        else:
            self.letter_label.setText(prediction)

        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        h,w,ch=frame.shape

        img=QImage(frame.data,w,h,ch*w,QImage.Format_RGB888)

        self.camera_label.setPixmap(QPixmap.fromImage(img))


    def closeEvent(self,event):

        self.cap.release()
        event.accept()


if __name__=="__main__":

    app=QApplication(sys.argv)

    window=SignLanguageApp()
    window.show()

    sys.exit(app.exec_())
import pyttsx3
import speech_recognition as sr
from tkinter import Tk

import UI_SimpleDialog as simple_dialog


speaker = pyttsx3.init()


def assistant_speak(audio):
    voices = speaker.getProperty("voices")
    if len(voices) > 1:
        speaker.setProperty("voice", voices[1].id)
    speaker.say(audio)
    speaker.runAndWait()
    return "Lynhz: " + audio + "\n"


def Mono_Speak(audio):
    """Backward-compatible wrapper for existing callers."""
    return assistant_speak(audio)


def Command():
    recognizer = sr.Recognizer()
    transcript = assistant_speak("Xin chào. Tôi là trợ lý ảo Lynhz.")
    transcript += assistant_speak("Bạn cần tôi hỗ trợ điều gì?")
    transcript += assistant_speak("Hãy nói hoặc nhập lệnh, tôi đang lắng nghe.")

    with sr.Microphone() as source:
        audio = recognizer.record(source, duration=4)

    try:
        query = recognizer.recognize_google(audio, language="vi")
        transcript += "\nUser: " + query
    except Exception:
        assistant_speak("Xin lỗi, tôi không nghe rõ. Vui lòng nói lại hoặc gõ lệnh.")
        root = Tk()
        app = simple_dialog.SimpleDialog(root)
        transcript += "\nUser: " + app.set_command()

    return transcript

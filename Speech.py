import gtts
from playsound import playsound


def speak(sentence: str):
    speech = gtts.gTTS(text=sentence, lang='en', slow=False)
    speech.save("speech_files/speech.mp3")
    playsound("speech_files/speech.mp3")
    return

from gtts import gTTS


def save_audio(text, language, filename):
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(filename)


if __name__ == '__main__':
    text = "Привет!!!"
    language = "ru"

    filename = 'speech.mp3'
    save_audio(text, language, filename)

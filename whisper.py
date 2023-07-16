import os
import openai

openai.api_key = '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'

audio_file= open("Porsche (with Doug DeMuro).mp3", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)

from pydub import AudioSegment

sound = AudioSegment.from_mp3("Porsche (with Doug DeMuro).mp3")

# PyDub handles time in milliseconds
ten_minutes = 10 * 60 * 1000

#將音檔分割成多個檔案

for i,chunk in enumerate(sound[::ten_minutes]):
    with open("Porsche (with Doug DeMuro)"+str(i)+".mp3", "wb") as f:
        chunk.export(f, format="mp3")

import yt_dlp as ydl

opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'outtmpl': '%(title)s.%(ext)s',
}

url = 'https://www.youtube.com/watch?v=V_fYfdXpkx4&list=PLjZkFWu3rWSE2cZ8L2CbiRMmHtJeF0kHh'

with ydl.YoutubeDL(opts) as ydl:
    ydl.download([url])
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
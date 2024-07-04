import requests

url = 'http://127.0.0.1:5000/transcribe'
files = {'file': open('hola.mp3', 'rb')}
response = requests.post(url, files=files)

print(response.json())
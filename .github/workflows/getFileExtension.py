import requests

response = requests.head("https://github.com/neuroneural/DeepCSR-fork/blame/master/src/utils.py")
print(response.headers['Content-Type'])
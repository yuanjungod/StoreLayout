import base64
import requests

with open("/Users/quantum/Downloads/u=368725982,2532668121&fm=27&gp=0.jpg", "rb") as f:
    # b64encode是编码，b64decode是解码
    base64_data = base64.b64encode(f.read())
    # base64.b64decode(base64data)
    print(base64_data)
    result = requests.post("http://ai-api.keruyun.com:5001/face_detect",
                           data={'base64_image_str': base64_data, 'appid': 2})
    print(result.text)


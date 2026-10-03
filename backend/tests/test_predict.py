import urllib.request
import io
import json
import numpy as np
import cv2

def test_api():
    # Create synthetic skin patch
    img = np.full((300, 300, 3), [140, 160, 220], dtype=np.uint8)
    cv2.circle(img, (150, 150), 35, (70, 80, 150), -1)
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    _, encoded = cv2.imencode('.jpg', img)

    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    body = io.BytesIO()
    body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="test.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode('utf-8'))
    body.write(encoded.tobytes())
    body.write(f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="modality"\r\n\r\nmobile_spacer\r\n'.encode('utf-8'))
    body.write(f'--{boundary}--\r\n'.encode('utf-8'))

    req = urllib.request.Request(
        'http://localhost:8000/api/v1/predict',
        data=body.getvalue(),
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
    )
    res = urllib.request.urlopen(req)
    data = json.loads(res.read().decode())
    print('Predict Status:', res.status)
    print('Prediction:', data['prediction'])
    print('Top 3:', data['top3'])
    print('Advice:', data['advice'])
    print('Quality:', data['quality'])
    print('Latency ms:', data['latency_ms'])

if __name__ == '__main__':
    test_api()

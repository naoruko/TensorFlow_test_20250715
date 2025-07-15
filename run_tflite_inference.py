import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# ラベルの読み込み
with open('label.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# TFLiteモデルの読み込み
interpreter = tflite.Interpreter(model_path='model_compat.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 入力サイズを取得
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

def preprocess(frame):
    resized = cv2.resize(frame, (width, height))
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

# カメラ起動
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess(frame)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)
    predicted_label = labels[predicted_index]
    confidence = output_data[0][predicted_index]

    # 結果表示
    text = f'{predicted_label} ({confidence:.2f})'
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
    cv2.imshow('TFLite Inference', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

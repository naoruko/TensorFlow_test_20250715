import tensorflow as tf
import numpy as np
import cv2

# モデルとラベル読み込み
model = tf.keras.models.load_model('model_transfer_learning.h5')

with open('label.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 推論用の前処理関数（モデルに合わせて調整）
def preprocess(frame):
    resized = cv2.resize(frame, (224, 224))  # モデルの入力サイズに合わせる
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)

# カメラ起動（USBカメラ or Piカメラ）
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 推論
    input_tensor = preprocess(frame)
    predictions = model.predict(input_tensor)
    predicted_index = np.argmax(predictions)
    predicted_label = labels[predicted_index]
    confidence = predictions[0][predicted_index]

    # 結果表示
    text = f'{predicted_label} ({confidence:.2f})'
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

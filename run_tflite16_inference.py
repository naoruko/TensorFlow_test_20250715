import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

# ------------------------------
# ✅ モデルとラベルの読み込み
# ------------------------------
MODEL_PATH = "model_float16.tflite"  # 変換した.tfliteモデル
LABEL_PATH = "label.txt"             # 各クラスのラベル

# ラベル読み込み
with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Interpreterのセットアップ
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 入力サイズ（例：96×96×3）
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# ------------------------------
# ✅ カメラ設定（解像度低め）
# ------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

print("💡 推論開始（qキーで終了）")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # 画像のリサイズと前処理
    img = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(img, axis=0).astype(np.float32) / 255.0  # 正規化

    # 推論開始
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    elapsed_time = time.time() - start_time

    # 結果表示
    prediction = np.squeeze(output_data)
    top_index = np.argmax(prediction)
    confidence = prediction[top_index]

    label = labels[top_index]
    text = f"{label} ({confidence*100:.1f}%)"
    print(f"[{text}]  推論時間: {elapsed_time*1000:.1f}ms")

    # 画面に表示（必要なら）
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('TFLite Image Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

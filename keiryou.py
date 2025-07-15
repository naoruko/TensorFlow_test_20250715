import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# モデルとラベルファイルの読み込み
interpreter = tflite.Interpreter(model_path="model_transfer_learning.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ラベル読み込み
with open("label.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# カメラ初期化（libcamera系OSでは CAP_V4L2 が安定）
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

print("📷 カメラ起動しました。'q' で終了します。")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ カメラから画像が取得できません")
        continue

    # 推論用画像の前処理
    input_shape = input_details[0]['shape']
    img = cv2.resize(frame, (input_shape[2], input_shape[1]))
    input_data = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

    # 推論実行
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    result = np.squeeze(output_data)

    # 最も高いスコアのクラスを取得
    top_index = np.argmax(result)
    label = labels[top_index]
    confidence = result[top_index]

    # 結果表示
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

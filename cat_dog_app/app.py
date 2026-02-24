import os
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# 업로드된 이미지를 저장할 폴더 설정
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 모델 로드 (앱 실행 시 한 번만 메모리에 올립니다)
print("모델을 불러오는 중입니다...")
model = tf.keras.models.load_model('best_model_xception.keras')
print("모델 로드 완료!")

def predict_image(img_path):
    # 1. 모델 입력 크기에 맞춰 이미지 로드 (150x150)
    img = image.load_img(img_path, target_size=(150, 150))
    
    # 2. 이미지를 배열로 변환
    img_array = image.img_to_array(img)
    
    # 3. 배치 차원 추가 (모델은 항상 배치를 받으므로 [1, 150, 150, 3] 형태로 만듦)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 4. 정규화 (보통 학습 시 0~1 사이로 정규화하므로 255로 나눕니다)
    img_array /= 255.0 

    # 5. 예측 수행
    prediction = model.predict(img_array)
    
    # 6. 결과 반환 (모델의 출력층 구성에 따라 이진 분류 결과를 해석합니다)
    # 일반적인 Keras 이진 분류의 경우 0.5를 기준으로 판별합니다.
    # (학습 코드를 모르는 상태이므로, 만약 결과가 반대라면 '개'와 '고양이'를 바꿔주세요)
    score = prediction[0][0]
    if score > 0.5:
        return "개 (Dog)", round(float(score) * 100, 2)
    else:
        return "고양이 (Cat)", round((1 - float(score)) * 100, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    image_url = None

    if request.method == 'POST':
        # 폼에 파일이 없는 경우 처리
        if 'file' not in request.files:
            return render_template('index.html', error="파일이 없습니다.")
        
        file = request.files['file']
        
        # 파일 이름이 비어있는 경우 처리
        if file.filename == '':
            return render_template('index.html', error="선택된 파일이 없습니다.")
            
        if file:
            # 파일 저장
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # 예측 함수 호출
            label, confidence = predict_image(filepath)
            prediction_text = f"이 사진은 {label}입니다. (확률: {confidence}%)"
            
            # 웹에 표시하기 위해 경로 수정 (슬래시 보장)
            image_url = f"/static/uploads/{file.filename}"

    return render_template('index.html', prediction_text=prediction_text, image_url=image_url)

if __name__ == '__main__':
    # 웹 서버 실행 (포트 5000)
    app.run(host='0.0.0.0', port=5000, debug=True)
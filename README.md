# 💊 SMU ML Project – 약제 분석 AI 웹 플랫폼

자연어와 이미지 기반 약제 분석을 제공하는 **FastAPI 기반 AI 플랫폼**입니다. 총 3가지 AI 모델을 통해 **알약 분류 → 부작용 여부 판단 → 질병 예측**까지 자동화된 분석을 제공합니다.

---

## 📌 프로젝트 개요

| 단계  | 모델 기능                  | 설명                                |
| --- | ---------------------- | --------------------------------- |
| 1️⃣ | **알약 이미지 분류**          | 사용자가 업로드한 이미지로 약 종류 시공            |
| 2️⃣ | **부작용 유무 분류 (Yes/No)** | 시공된 약의 부작용 저작 유무 판단               |
| 3️⃣ | **중복 약제 기반 질병 분류**     | 보건자가 보곱 중인 약제 정보를 통해 가능성 있는 질병 예측 |

---

## 💻 주요 특징

* ✅ **FastAPI 기반 건강한 REST API 서버**로 빠른 응답성
* ✅ Swagger UI 를 통한 간편한 API 테스트
* ✅ AI 모델 3종 통합: 이미지 인식 + 이지리 분류 + 단수 분류
* ✅ 사용자 입력만으로 질병 예측 결과 기본 보고서 제공

---

## 🛠기술 스택

* **Backend:** FastAPI, Uvicorn
* **AI Model:** PyTorch or TensorFlow 기반 디플렉니드 모델 3종
* **Image Processing:** OpenCV, PIL
* **Data Handling:** Pandas, NumPy

---

## 🚀 설치 및 실행

```bash
# 1. 레포지트리 클론
git clone https://github.com/YnexKimdoyeon/SMU_ML_PROJECT.git
cd SMU_ML_PROJECT

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 서버 실행
uvicorn main:app --reload
```

---

## 🔗 API 구조

### `POST /predict-pill`

* 입력: 알약 이미지
* 결과: 알약 종류 (ex. 건티린, 캐프시리모)

### `POST /check-adverse-effect`

* 입력: 알약 ID 또는 이름
* 결과: Yes / No (부작용 유무)

### `POST /predict-disease`

* 입력: 보곱 중인 약제 목록
* 결과: 관련 가능 질병 예측

---

## 📊 예시 결과

| 입력 조건               | 알약 분류   | 부작용 결과 | 질병 예측    |
| ------------------- | ------- | ------ | -------- |
| 이미지: test\_pill.jpg | 캐프시리모   | No     | 건너지불지화   |
| 약 ID: 1234          | 아이브푸르퍼널 | Yes    | 심리건강 가능성 |

---

## 📊 회고 & 활용 계획

* 지원 약 데이터 복고 및 진행 기본
* OCR 기능 가운 다운 알약 문자 인식 기능 규포
* 개인 데이터 기본 질병 분석 결함
* 메일 게임을 통한 통화적 진단 및 기억 관련 서비스 업그레이드 계획

---


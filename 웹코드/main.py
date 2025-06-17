from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
<<<<<<< HEAD
from pydantic import BaseModel
import traceback
=======

>>>>>>> 964dc5a7d7a5e8f0def33379fa5471b5cd6382ed
import cv2
import numpy as np
import os
import uuid
<<<<<<< HEAD
from PIL import Image
from pathlib import Path
from pill_classifier import *
from get_cli_args import get_cli_args
from joblib import load
import json
import shutil
import pandas as pd
from itertools import combinations
import math
import logging

# 전역 변수로 모델 선언
model = None
le1 = None
le2 = None
pill_classifier_args = None

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global model, le1, le2, pill_classifier_args
    try:
        # 부작용 예측 모델 로드
        model_data = load("static\\model\\First_Model\\model\\full_model.pkl", mmap_mode=None)
        model = model_data['model']
        le1 = model_data['le1']
        le2 = model_data['le2']
        print("부작용 예측 모델 로드 완료")
    except Exception as e:
        print("부작용 예측 모델 로드 실패:")
        traceback.print_exc()
        model = None
        le1 = None
        le2 = None

    try:
        # 알약 분류 모델 초기화
        job = 'resnet152'
        pill_classifier_args = get_cli_args(job=job, run_phase='test', aug_level=0, dataclass='01')
        print("알약 분류 모델 초기화 완료")
    except Exception as e:
        print("알약 분류 모델 초기화 실패:")
        traceback.print_exc()
        pill_classifier_args = None

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output", StaticFiles(directory="output"), name="output")
=======

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
>>>>>>> 964dc5a7d7a5e8f0def33379fa5471b5cd6382ed
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

<<<<<<< HEAD
def preprocess_image(image):
    try:
        if image is None or image.size == 0:
            raise ValueError("유효하지 않은 이미지입니다.")
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        bg_removed = 255 - cv2.absdiff(gray, dilated)

        blur = cv2.GaussianBlur(bg_removed, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 4
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return morph
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {str(e)}")
        raise
=======

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg_removed = 255 - cv2.absdiff(gray, dilated)

    blur = cv2.GaussianBlur(bg_removed, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morph

>>>>>>> 964dc5a7d7a5e8f0def33379fa5471b5cd6382ed

def is_valid_contour(contour, w, h, area, mask_roi):
    aspect_ratio = w / h
    perimeter = cv2.arcLength(contour, True)

<<<<<<< HEAD
=======
    # 기본 조건: 크기, 종횡비, 윤곽 길이
>>>>>>> 964dc5a7d7a5e8f0def33379fa5471b5cd6382ed
    if area < 1000 or area > 60000:
        return False
    if aspect_ratio < 0.3 or aspect_ratio > 3.5:
        return False
    if perimeter < 100:
        return False

<<<<<<< HEAD
=======
    # 흰 배경 비율이 너무 높으면 '글씨만 있는 객체'로 판단
>>>>>>> 964dc5a7d7a5e8f0def33379fa5471b5cd6382ed
    white_ratio = cv2.countNonZero(mask_roi) / (w * h)
    if white_ratio > 0.75:
        return False

    return True
<<<<<<< HEAD
def replace_white_with_brown(image, threshold=240):
    """
    흰색(또는 거의 흰색) 배경을 어두운 갈색(20, 9, 20)으로 치환합니다.
    """
    brown_color = (20, 9, 20)  # BGR

    # 흰색 마스크 만들기
    white_mask = cv2.inRange(image, (threshold, threshold, threshold), (255, 255, 255))

    # 갈색으로 치환
    image[white_mask == 255] = brown_color
    return image
def extract_square_image(image, x, y, w, h, bg_color=(20, 9, 20)):
    # 알약 영역 자르기
    pill_crop = image[y:y+h, x:x+w]
    size = max(w, h)
    background = np.full((size, size, 3), bg_color, dtype=np.uint8)

    # 마스크 생성: 알약 경계 추출용
    gray = cv2.cvtColor(pill_crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  # 거의 흰색을 제거
    mask = cv2.medianBlur(mask, 5)

    # 중심 위치 계산
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    # 알약 영역만 복사 (흰색 테두리 제거한 영역)
    roi = background[y_offset:y_offset+h, x_offset:x_offset+w]
    pill_region = cv2.bitwise_and(pill_crop, pill_crop, mask=mask)
    bg_region = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    combined = cv2.add(pill_region, bg_region)

    # 배경에 삽입
    background[y_offset:y_offset+h, x_offset:x_offset+w] = combined

    return cv2.resize(background, (224, 224))
=======


def extract_square_image(image, x, y, w, h):
    crop = image[y:y + h, x:x + w]
    square_size = max(w, h)
    square_img = 255 * np.ones((square_size, square_size, 3), dtype=np.uint8)
    x_offset = (square_size - w) // 2
    y_offset = (square_size - h) // 2
    square_img[y_offset:y_offset + h, x_offset:x_offset + w] = crop
    resized = cv2.resize(square_img, (128, 128))
    return resized


>>>>>>> 964dc5a7d7a5e8f0def33379fa5471b5cd6382ed
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

<<<<<<< HEAD
def clean_json(obj):
    """재귀적으로 NaN, inf, -inf를 None으로 변환"""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            logging.warning(f"JSON 변환 불가 값 발견: {obj}")
            return None
        return obj
    else:
        return obj

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # 업로드 시작 전에 output 폴더 비우기
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        image = replace_white_with_brown(image)
        if image is None:
            return JSONResponse(status_code=400, content={"message": "이미지 읽기 실패"})

        processed = preprocess_image(image)
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        saved_count = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            mask_roi = processed[y:y + h, x:x + w]

            if not is_valid_contour(contour, w, h, area, mask_roi):
                continue

            pill_img = extract_square_image(image, x, y, w, h)
            file_name = f"{uuid.uuid4().hex[:8]}.png"
            save_path = os.path.join(OUTPUT_DIR, file_name)
            cv2.imwrite(save_path, pill_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            saved_count += 1

        if saved_count == 0:
            return JSONResponse(
                status_code=400,
                content={"message": "이미지에서 알약을 찾을 수 없습니다."}
            )

        # main_cls01_dir.py 방식으로 분류 및 결과 출력
        pill_classifier_args.dataset_valid = Dataset_Dir(
            pill_classifier_args,
            OUTPUT_DIR,
            transform=transform_normalize,
            run_phase='test'
        )
        pill_classifier_args.batch_size = len(pill_classifier_args.dataset_valid)
        pill_classifier_args.verbose = False
        pill_classifier(pill_classifier_args)

        # 파일명과 class_id 저장
        results = []
        class_ids = []
        for img_name, pred in zip(pill_classifier_args.dataset_valid.list_images, pill_classifier_args.list_preds):
            results.append({"file": img_name, "class_id": pred})
            class_ids.append(pred)

        # CSV에서 class_id와 dl_material_en 매핑
        csv_path = "pill_metadata_sampled_with_class_firstcol.csv"
        df = pd.read_csv(csv_path)
        # A열이 class_id, dl_material_en이 매핑값이라고 가정
        classid_to_dl = {}
        for _, row in df.iterrows():
            classid_to_dl[row.iloc[0]] = row['dl_material_en']

        # 분류된 class_id → dl_material_en 리스트 (|로 나눠서 strip)
        dl_materials = []
        for cid in class_ids:
            dl = classid_to_dl.get(cid)
            if dl:
                # '|'로 나눠서 strip 후 모두 추가
                for item in str(dl).split('|'):
                    item = item.strip()
                    if item:
                        dl_materials.append(item)

        # 모든 쌍 조합 생성 및 부작용 예측
        side_effect_results = []
        for d1, d2 in combinations(dl_materials, 2):
            try:
                pred = None
                if model and le1 is not None and le2 is not None:
                    d1_enc = le1.transform([d1])[0]
                    d2_enc = le2.transform([d2])[0]
                    pred = model.predict([[d1_enc, d2_enc]])[0]
                # NaN, inf 체크
                if pred is not None and (isinstance(pred, float) and (math.isnan(pred) or math.isinf(pred))):
                    pred = None
                side_effect_results.append({
                    "drug1": d1,
                    "drug2": d2,
                    "side_effect": bool(pred) if pred is not None else False,
                    "message": "예측된 부작용 있음" if pred == 1 else "예측된 부작용 없음"
                })
            except Exception as e:
                side_effect_results.append({
                    "drug1": d1,
                    "drug2": d2,
                    "side_effect": False,
                    "message": "예측된 부작용 없음"
                })

        # 반환 직전 결과 클린징
        safe_results = clean_json(results)
        safe_side_effects = clean_json(side_effect_results)


        return {
            "message": f"{saved_count}개의 알약이 저장되었습니다.",
            "results": safe_results,
            "side_effects": safe_side_effects
        }

    except Exception as e:
        import traceback
        logging.error("업로드 처리 중 예외 발생:\n" + traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"message": f"처리 중 오류가 발생했습니다: {str(e)}"}
        )

class DrugPair(BaseModel):
    drug1: str
    drug2: str

class Dataset_Dir(Dataset):
    def __init__(self, args, dir_dataset, transform=None, target_transform=None, run_phase='train'):
        self.args = args
        self.dir_dataset = dir_dataset
        self.transform = transform
        self.target_transform = target_transform

        self.list_images = [ png.name  for png in Path(dir_dataset).iterdir() if png.suffix == '.png']
        self.run_phase = run_phase

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.dir_dataset, self.list_images[idx])).convert('RGB')  # <-- 중요
        label = 0
        path_img = self.list_images[idx]
        aug_name = ""
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.run_phase == 'valid' or self.run_phase == 'test':
            return image, label, path_img, aug_name
        else:
            return image, label

@app.post("/predict-side-effect")
async def predict_side_effect(drug_pair: DrugPair):
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"message": "모델이 로드되지 않았습니다."}
        )

    try:
        d1 = le1.transform([drug_pair.drug1])[0]
        d2 = le2.transform([drug_pair.drug2])[0]
        pred = model.predict([[d1, d2]])[0]
        
        return {
            "drug1": drug_pair.drug1,
            "drug2": drug_pair.drug2,
            "side_effect": bool(pred),
            "message": "예측된 부작용 있음" if pred == 1 else "예측된 부작용 없음"
        }
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"message": "입력한 약물이 인코더에 없습니다."}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"예측 중 오류가 발생했습니다: {str(e)}"}
        )
    
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
=======

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(status_code=400, content={"message": "이미지 읽기 실패"})

    processed = preprocess_image(image)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    saved_count = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        mask_roi = processed[y:y + h, x:x + w]

        if not is_valid_contour(contour, w, h, area, mask_roi):
            continue

        pill_img = extract_square_image(image, x, y, w, h)
        file_name = f"{uuid.uuid4().hex[:8]}.jpg"
        save_path = os.path.join(OUTPUT_DIR, file_name)
        cv2.imwrite(save_path, pill_img)

        results.append({
            "file": file_name
        })
        saved_count += 1

    return {
        "message": f"{saved_count}개의 알약이 저장되었습니다.",
        "results": results
    }
>>>>>>> 964dc5a7d7a5e8f0def33379fa5471b5cd6382ed

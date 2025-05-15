from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
import os
import uuid

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
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


def is_valid_contour(contour, w, h, area, mask_roi):
    aspect_ratio = w / h
    perimeter = cv2.arcLength(contour, True)

    # 기본 조건: 크기, 종횡비, 윤곽 길이
    if area < 1000 or area > 60000:
        return False
    if aspect_ratio < 0.3 or aspect_ratio > 3.5:
        return False
    if perimeter < 100:
        return False

    # 흰 배경 비율이 너무 높으면 '글씨만 있는 객체'로 판단
    white_ratio = cv2.countNonZero(mask_roi) / (w * h)
    if white_ratio > 0.75:
        return False

    return True


def extract_square_image(image, x, y, w, h):
    crop = image[y:y + h, x:x + w]
    square_size = max(w, h)
    square_img = 255 * np.ones((square_size, square_size, 3), dtype=np.uint8)
    x_offset = (square_size - w) // 2
    y_offset = (square_size - h) // 2
    square_img[y_offset:y_offset + h, x_offset:x_offset + w] = crop
    resized = cv2.resize(square_img, (128, 128))
    return resized


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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

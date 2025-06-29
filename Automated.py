from ultralytics import YOLO
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
import argparse

# 입력 이미지 및 모델
def __main__():

    image_path = "https://static.wixstatic.com/media/0716ed_ba85d54a7dfa4e76882762a8ab8e2256~mv2_d_3189_2362_s_2.jpg/v1/fill/w_703,h_522,al_c,q_85,usm_0.66_1.00_0.01,enc_avif,quality_auto/0716ed_ba85d54a7dfa4e76882762a8ab8e2256~mv2_d_3189_2362_s_2.jpg"
    model = YOLO("yolov8m-seg-speech-bubble.pt")
    font_path = "ko.otf"
    text_to_insert = {
        'Doctor':"",
        'Patient':"",
    }
    parser = argparse.ArgumentParser(description="옵션 인자 짧은 이름과 긴 이름 예제")

    parser.add_argument('--doc','-d', type=str, help='의사가 하는 말')
    parser.add_argument('--patient','-p', type=int, help='듀얼러가 하는 말')
    parser.add_argument('--font','-f', type=int, help='폰트 경로')


    args = parser.parse_args()

    text_to_insert['Doctor'] = args.doc
    text_to_insert['Patient'] = args.patient
    font_path = args.font
    print(f"이름: {args.name}")
    print(f"나이: {args.age}")
    # 이미지 불러오기
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    # 예측 실행
    results = model.predict(source=image_path, save=False, imgsz=max(h, w))

    for r in results:
        if r.masks is None:
            continue

        class_names = model.names
        target_class_name = "speech bubble"

        # 마스크, 클래스 ID, confidence 모두 순회
        for i, (mask, cls_id, conf, box) in enumerate(zip(r.masks.data, r.boxes.cls, r.boxes.conf, r.boxes.xyxy)):
            class_id = int(cls_id.item())
            if class_names[class_id] != target_class_name:
                continue

            # 박스
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_names[class_id]} {conf.item():.2f}"
            cv2.putText(image_np, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)

            # 마스크 윤곽선 추출
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            binary = (mask_resized > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                contour = contour.astype(np.float32)
                contour -= [cx, cy]
                contour *= 0.95
                contour += [cx, cy]
                contour = contour.astype(np.int32)

                # 하얀색으로 채움 (지우기)
                cv2.drawContours(image_np, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

                # 텍스트 그릴 공간 계산
                x, y, w_box, h_box = cv2.boundingRect(contour)
                pil_img = Image.fromarray(image_np)
                draw = ImageDraw.Draw(pil_img)

                # 폰트 크기 자동 조정
                font_size = 30
                while font_size > 10:
                    font = ImageFont.truetype(font_path, font_size)
                    bbox = draw.multiline_textbbox((0, 0), text_to_insert[i], font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    tx = x + (w_box - text_w) // 2
                    ty = y + (h_box - text_h) // 2

                    # contour 내부 포함 검사
                    all_inside = True
                    for dy in range(0, text_h, 3):  # 성능 위해 3픽셀 간격 점검
                        for dx in range(0, text_w, 3):
                            px = tx + dx
                            py = ty + dy
                            if cv2.pointPolygonTest(contour, (px, py), measureDist=False) < 0:
                                all_inside = False
                                break
                        if not all_inside:
                            break

                    if all_inside:
                        break
                    font_size -= 1

                # 텍스트 중앙 삽입
                tx = x + (w_box - text_w) // 2
                ty = y + (h_box - text_h) // 2

                draw.multiline_text((tx, ty), text_to_insert[i], font=font, fill=(0, 0, 0), align="center")
                image_np = np.array(pil_img)
    output_path = "result.png"
    Image.fromarray(image_np).save(output_path)
    print(f"이미지가 {output_path}로 저장되었습니다.")


# 저장 및 표시

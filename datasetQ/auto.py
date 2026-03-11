import cv2
import os
import torch
import time
from l2cs import Pipeline, render

# 1. Tạo thư mục lưu dữ liệu
save_dir = "dataset_auto"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 2. Khởi tạo bộ não AI L2CS-Net
print("Đang khởi động AI L2CS-Net...")
gaze_pipeline = Pipeline(
    weights='models/L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=torch.device('cpu') # Bạn có thể đổi thành 'cuda' nếu có GPU
)

# 3. Mở camera nét và lật ảnh
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow("Auto Annotation Tool", cv2.WINDOW_NORMAL)

print("\n=== TOOL TỰ ĐỘNG GÁN NHÃN BẰNG AI ===")
print("- Cứ mỗi 1 giây, AI sẽ tự phán đoán và lưu 1 ảnh.")
print("- Hãy diễn cảnh tập trung (0) hoặc quay cóp (1).")
print("- Bấm phím 'q' để kết thúc.\n")

stt = 1
last_save_time = time.time()
save_interval = 1.0 # Thời gian nghỉ giữa 2 lần chụp (1.0 giây)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    
    # AI phân tích khung hình
    results = gaze_pipeline.step(frame)
    
    # Tạo một bản copy để vẽ đồ họa lên màn hình hiển thị
    # (KHÔNG vẽ thẳng lên frame gốc, vì ảnh dataset nộp cho sếp phải là ảnh sạch)
    display_frame = frame.copy()
    display_frame = render(display_frame, results)
    
    # KIỂM TRA LOGIC VÀ TỰ ĐỘNG GÁN NHÃN
    if results.pitch is not None and len(results.pitch) > 0:
        pitch = results.pitch[0]
        yaw = results.yaw[0]
        
        # Đặt ngưỡng gian lận (Ví dụ: góc lệch lớn hơn 0.35 radian ~ khoảng 20 độ)
        threshold = 0.35 
        
        if abs(pitch) > threshold or abs(yaw) > threshold:
            label = 1 # Gian lận (Positive)
            status_text = "GIAN LAN (1)"
            color = (0, 0, 255) # Đỏ
        else:
            label = 0 # Bình thường (Negative)
            status_text = "BINH THUONG (0)"
            color = (0, 255, 0) # Xanh lá
            
        # Ghi chữ lên màn hình để bạn biết AI đang nghĩ gì
        cv2.putText(display_frame, f"AI Doan: {status_text}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Tự động lưu ảnh sau mỗi `save_interval` giây
        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            # Lưu ý quan trọng: Lưu frame GỐC (không có điểm đỏ/chữ)
            filename = f"{stt:03d}_{label}.png"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)
            
            print(f"[{status_text}] Đã tự động lưu: {filename}")
            stt += 1
            last_save_time = current_time

    # Hiển thị lên màn hình
    cv2.imshow("Auto Annotation Tool", display_frame)
    
    # Bấm 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Đã hoàn thành! Thu thập được {stt - 1} ảnh vào thư mục {save_dir}.")
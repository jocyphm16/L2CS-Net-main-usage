import argparse
import cv2
import torch
from l2cs import Pipeline, render

def parse_args():
    parser = argparse.ArgumentParser(description="Gaze estimation on video file")
    parser.add_argument('--video-path', type=str, required=True, help='Đường dẫn tới video đầu vào')
    parser.add_argument('--snapshot', type=str, default='models/L2CSNet_gaze360.pkl', help='Đường dẫn tới file trọng số model')
    parser.add_argument('--device', type=str, default='cpu', help='Sử dụng cpu hoặc gpu')
    parser.add_argument('--output-path', type=str, default='output.mp4', help='Tên file video đầu ra')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Đang khởi tạo mô hình L2CS-Net...")
    # Khởi tạo Pipeline (giống như trong demo.py)
    gaze_pipeline = Pipeline(
        weights=args.snapshot,
        arch='ResNet50',
        device=torch.device(args.device)
    )
    
    # Mở video đầu vào
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video tại {args.video_path}")
        return
        
    # Lấy các thông số của video gốc để tạo video đầu ra (width, height, fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Khởi tạo công cụ ghi video của OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Định dạng mp4
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))
    
    print(f"Bắt đầu xử lý video: {args.video_path} ({total_frames} frames)")
    
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Lưu ý: Không dùng cv2.flip() ở đây vì video quay sẵn thường đã đúng chiều
        
        # Đưa frame qua mạng nơ-ron để lấy kết quả
        results = gaze_pipeline.step(frame)
        
        # Vẽ bounding box, vector và ĐIỂM ĐỎ (hàm render gọi tới vis.py của bạn)
        frame = render(frame, results)
        
        # Ghi frame đã vẽ vào file output
        out.write(frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Đã xử lý {frame_count}/{total_frames} frames...")
            
    # Giải phóng bộ nhớ và đóng file
    cap.release()
    out.release()
    print(f"Hoàn thành! Video đã được lưu tại: {args.output_path}")

if __name__ == '__main__':
    main()
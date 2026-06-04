import os
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO, YOLOv10

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def profile_model(model_path, is_v10=False, dataset_path='coco/images/val2017'):
    if is_v10:
        model = YOLOv10(model_path)
    else:
        model = YOLO(model_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f'載入{model_path},運算裝置{device.upper()}')

    pure_inference_time = 0.0
    pure_postprocess_time = 0.0
    total_images = 0

    print(f"[*] 正在對資料集進行端到端推論...")

    # 加上 stream=True 防爆記憶體，並用 for 迴圈迭代抽取出每一張圖的結果
    results_generator = model.predict(source=dataset_path, imgsz=640, verbose=False, stream=True)

    for r in results_generator:
        # 逐張提取 Ultralytics 底層紀錄的純粹晶片時間
        pure_inference_time += r.speed['inference']
        pure_postprocess_time += r.speed['postprocess']
        total_images += 1

        # 每跑 500 張印一次進度
        if total_images % 200 == 0:
            torch.cuda.empty_cache()
        if total_images % 500 == 0 :
            print(f"    已完成 {total_images} 張圖片...")
        del r
    torch.cuda.empty_cache()

    avg_inf = pure_inference_time / total_images
    avg_post = pure_postprocess_time / total_images
    total_pure_latency = avg_inf + avg_post
    pure_fps = 1000 / total_pure_latency

    return avg_inf, avg_post, total_pure_latency, pure_fps

v8_model_name = 'yolov8m.pt'
v10_model_name = 'yolov10m.pt'
test_image = 'coco/images/val2017'

v8_inf, v8_post, v8_pure_latency,v8_pure_fps = profile_model(v8_model_name, is_v10=False, dataset_path=test_image)
v10_inf, v10_post, v10_pure_latency,v10_pure_fps = profile_model(v10_model_name, is_v10=True, dataset_path=test_image)

print(f"{'評測性能指標':<25}{'YOLOv8m (傳統 NMS)':<22}{'YOLOv10m (NMS-Free)':<22}")
print("-"*65)
print(f"{'網路推論時間 (Inference)':<25}{v8_inf:<22.2f} ms{v10_inf:<22.2f} ms")
print(f"{'後處理時間 (Postprocess)':<25}{v8_post:<22.2f} ms{v10_post:<22.2f} ms")
print(f"{'純總延遲 (Pure Latency)':<25}{v8_pure_latency:<22.2f} ms{v10_pure_latency:<22.2f} ms")
print(f"{'純每秒影格數 (Pure FPS)':<25}{v8_pure_fps:<22.2f}{v10_pure_fps:<22.2f}")

models = ['YOLOv8m (with NMS)','YOLOv10m (NMS-Free)']
fps_values = [v8_pure_fps, v10_pure_fps]
latency_values = [v8_pure_latency, v10_pure_latency]

#建立一個包含兩個子圖的畫布( 1列2行 )
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 圖一：Pure FPS 對比
bars1 = ax1.bar(models, fps_values, color=['#3498db', '#2ecc71'], width=0.5)
ax1.set_title('Pure Speed (FPS) - Higher is Better', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frames Per Second (FPS)')
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.bar_label(bars1, fmt='%.2f', padding=3)

# 圖二：Pure Latency 對比
bars2 = ax2.bar(models, latency_values, color=['#e74c3c', '#9b59b6'], width=0.5)
ax2.set_title('Pure Latency (ms) - Lower is Better', fontsize=11, fontweight='bold')
ax2.set_ylabel('Latency (ms)')
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.bar_label(bars2, fmt='%.2f', padding=3)

plt.tight_layout()
output_chart = "yolo_comparison_chart3.png"
plt.savefig(output_chart, dpi=300)
print(f"✅ 【成功】對比圖表已儲存至: {os.path.abspath(output_chart)}")
plt.show()






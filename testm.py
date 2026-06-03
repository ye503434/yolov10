
from ultralytics import YOLOv10

# 載入剛剛下載的權重
model = YOLOv10('yolov10m.pt')

# 執行預測
source = '../datasets/coco8/images/train'
results = model.predict(source=source, save=True)

# 輸出結果儲存的資料夾路徑
print("偵測完成！結果已儲存在：", results[0].save_dir)
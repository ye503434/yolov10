import os
from pathlib import Path
# 💡 修正：我們只保留真正需要的 download 函數進口
from ultralytics.utils.downloads import download

# 1. 強制設定下載到目前的 G 槽專案目錄下
datasets_dir = Path("./datasets")
coco_dir = datasets_dir / "coco"

print("[*] 步驟一：正在下載 YOLO 格式的 COCO 標籤檔...")
segments = False

# 💡 核心優化：直接把官方釋出的原始下載網址貼進來，完美避開 PyCharm 的 Reference 找不到問題！
if segments:
    labels_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip"
else:
    labels_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"

# 開始下載標籤
download([labels_url], dir=datasets_dir)

print("\n[*] 步驟二：正在下載 COCO 2017 驗證集圖片 (大約 1 GB)...")
img_urls = [
    "http://images.cocodataset.org/zips/val2017.zip"  # 1G, 5k images
]

# 開始下載圖片 (threads=3 代表開啟 3 線程平行加速下載)
download(img_urls, dir=coco_dir / "images", threads=3)

print("\n✅ 【大功告成】COCO 驗證資料集已成功下載並自動解壓！")
print(f"   圖片存放位置: {os.path.abspath(coco_dir / 'images' / 'val2017')}")
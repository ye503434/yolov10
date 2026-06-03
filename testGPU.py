import torch
print("GPU 是否可用：", torch.cuda.is_available())
print("GPU 型號名稱：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "無")
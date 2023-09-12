import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images
print(type(imgs))
print(imgs)

# Inference
results = model(imgs)
print("* " * 10)
print("model", "\n", model)
print("* " * 10)
print("results", type(results), "\n", results)
print("* " * 10)
print("results dir", dir(results))
print("* " * 10)

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)
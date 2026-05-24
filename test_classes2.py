import os
import json

data_dir = "data/raw/PlantVillage"
classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
print("Classes:", classes)
with open("models/classes.json", "w") as f:
    json.dump(classes, f)

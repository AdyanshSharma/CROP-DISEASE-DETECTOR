import torch
import pprint
state_dict = torch.load("models/best_model.pth", map_location="cpu")
print("Classifier weight shape:", state_dict["classifier.1.weight"].shape)

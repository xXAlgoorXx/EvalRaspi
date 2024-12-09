from hailoEval_utils import HailoCLIPImage, HailoCLIPText
from PIL import Image
import clip
import torch
import open_clip

def printEval(image_features, text_features):
    image_features = torch.Tensor(image_features)
    text_features = torch.Tensor(text_features)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity.topk(5)
    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{names2[index]:>16s}: {100 * value.item():.2f}%")

device = "cuda" if torch.cuda.is_available() else "cpu"
gemmLayerjson_path = "Evaluation/models/TinyClip19M/gemmLayer_TinyCLIP-ResNet-19M.json"
hef_path = 'Evaluation/models/TinyClip19M/TinyCLIP-ResNet-19M.hef'
onnx_path = "Evaluation/models/TinyClip19M/TinyCLIP-ResNet-19M.onnx"
model_name = "TinyCLIP-ResNet-19M-Text-19M"
hailoInference = HailoCLIPImage(hef_path,gemmLayerjson_path)
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    model_name,
    pretrained=f"tinyClipModels/{model_name}-LAION400M.pt"
)

names2 = ["architectural", "office", "residential", "school", "manufacturing",
            "cellar", "laboratory", "construction site", "mining", "tunnel"]

names2 = ["car","dog","cat","house","wrench","Blue"]

# Example usage
image_path = "Evaluation/testImg/pexels-mikebirdy-170811.jpg"
image_pre = Image.open(image_path)
image = preprocess_val(image_pre).unsqueeze(0).to(device)
#image = hailoInference.performPreprocess(image_pre).unsqueeze(0)
tokenizer = open_clip.get_tokenizer(model_name)
text_inputs = tokenizer(names2).to(device)
    
# CLIP
with torch.no_grad():
    imageEmb_clip = model.encode_image(image).detach().numpy().squeeze()
    text_features = model.encode_text(text_inputs)

# ONNX
session = ort.InferenceSession(onnx_path)
imageEmb_onnx = np.array(session.run(None, {"input": image.numpy()})).squeeze()
#imageEmb_onnx = hailoInference.performPostprocess(np.array(imageEmb_onnx).flatten())

# HAILO
imageEmb_hailo = hailoInference.encode_image(image)

# Print results
print("\n=== CLIP ===")
print(f"Shape: {imageEmb_clip.shape}")
printEval(imageEmb_clip,text_features)

print("\n=== ONNX ===")
print(f"Shape: {imageEmb_onnx.shape}")
diff = imageEmb_onnx - imageEmb_clip
print(f"Mean diff to CLIP: {np.mean(diff)}")
print(f"Var diff to CLIP: {np.var(diff)}")
printEval(imageEmb_onnx,text_features)

print("\n=== Hailo ===")
print(f"Shape: {imageEmb_hailo.shape}")
diff = imageEmb_onnx - imageEmb_hailo
print(f"Mean diff to CLIP: {np.mean(diff)}")
print(f"Var diff to CLIP: {np.var(diff)}")
printEval(imageEmb_hailo,text_features)
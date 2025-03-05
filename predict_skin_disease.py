import torch
import torchvision.transforms as transforms
from PIL import Image
from models.evi_cem import Evidential_CEM

# Load mô hình đã huấn luyện
ckpt_path = "checkpoints/last.ckpt"
model = Evidential_CEM.load_from_checkpoint(ckpt_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


# Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_skin_disease(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).cuda()

    with torch.no_grad():
        (alpha, beta), y_logits = model(image, None, train=False)

    # Dự đoán triệu chứng
    c_probs = alpha / (alpha + beta)
    concept_names = [
    "Vesicle", "Papule", "Macule", "Plaque", "Abscess", "Pustule", "Bulla", "Patch",
    "Nodule", "Ulcer", "Crust", "Erosion", "Excoriation", "Atrophy", "Exudate",
    "Purpura/Petechiae", "Fissure", "Induration", "Xerosis", "Telangiectasia",
    "Scale", "Scar", "Friable", "Sclerosis", "Pedunculated", "Exophytic/Fungating",
    "Warty/Papillomatous", "Dome-shaped", "Flat topped", "Brown(Hyperpigmentation)",
    "Translucent", "White(Hypopigmentation)", "Purple", "Yellow", "Black",
    "Erythema", "Comedo", "Lichenification", "Blue", "Umbilicated", "Poikiloderma",
    "Salmon", "Wheal", "Acuminate", "Burrow", "Gray", "Pigmented", "Cyst"
     ]

    detected_concepts = [concept_names[i] for i, prob in enumerate(c_probs.squeeze()) if prob > 0.5]

    # Dự đoán bệnh
    y_pred = torch.softmax(y_logits, dim=-1)
    disease_classes = ["Non-Neoplastic", "Benign", "Malignant"]  # Danh sách bệnh
    predicted_disease = disease_classes[y_pred.argmax().item()]

    return detected_concepts, predicted_disease

# Thử nghiệm với ảnh bệnh da liễu
image_path = "path/to/skin_image.jpg"
concepts, disease = predict_skin_disease(image_path)
print(f"Triệu chứng: {concepts}")
print(f"Chẩn đoán: {disease}")

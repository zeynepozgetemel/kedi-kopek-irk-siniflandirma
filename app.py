"""Gradio uygulaması: fotoğraf yükle -> 37 kedi/köpek ırkından en olası 5 tahmin.

Çalıştırma:  python app.py   (outputs/best_model.pt gerekli)
"""
import json, time
import gradio as gr
import timm, torch
from torchvision import transforms as T

ck = torch.load("outputs/best_model.pt", map_location="cpu")
CLASSES, SIZE = ck["classes"], ck["img_size"]
model = timm.create_model("resnet18", pretrained=False, num_classes=len(CLASSES))
model.load_state_dict(ck["state_dict"]); model.eval()

CATS = {"abyssinian", "bengal", "birman", "bombay", "british_shorthair", "egyptian_mau", "maine_coon",
        "persian", "ragdoll", "russian_blue", "siamese", "sphynx"}
def pretty(c):
    return ("🐱 " if c in CATS else "🐶 ") + c.replace("_", " ").title()

tf = T.Compose([T.Resize(int(SIZE * 256 / 224)), T.CenterCrop(SIZE), T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def predict(img):
    if img is None:
        return {}, ""
    x = tf(img.convert("RGB")).unsqueeze(0)
    t = time.perf_counter()
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0]
    ms = (time.perf_counter() - t) * 1000
    top = torch.topk(p, 5)
    res = {pretty(CLASSES[i]): float(v) for v, i in zip(top.values, top.indices)}
    best = float(top.values[0])
    note = f"Tahmin süresi: {ms:.1f} ms"
    if best < 0.35:
        note += " — ⚠️ Model emin değil (güven < %35). Hayvan bu 37 ırktan biri olmayabilir."
    return res, note

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Kedi veya köpek fotoğrafı"),
    outputs=[gr.Label(num_top_classes=5, label="Tahmin (Top-5)"), gr.Textbox(label="Bilgi")],
    title="Kedi & Köpek Irkı Sınıflandırıcı",
    description="ResNet-18 (transfer learning) · Oxford-IIIT Pet · 37 ırk (12 kedi, 25 köpek). "
                "Bir fotoğraf yükleyin veya kameradan çekin.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()

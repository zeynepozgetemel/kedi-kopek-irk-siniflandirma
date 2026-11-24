"""Kendi çektiğiniz fotoğrafları modele verip sonuçları tek bir görselde toplar.

Kullanım:  python predict_photos.py my_photos/*.jpg --truth "british_shorthair,bombay,persian,pug,?"
--truth: her fotoğrafın gerçek ırkı (bilinmiyorsa ?), virgülle ve fotoğraf sırasıyla.
Çıktı: outputs/my_photos_results.png ve outputs/my_photos_results.json
"""
import argparse, json, math, time
import timm, torch
from PIL import Image, ImageOps
from torchvision import transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("images", nargs="+")
ap.add_argument("--truth", default="")
ap.add_argument("--model", default="outputs/best_model.pt")
a = ap.parse_args()
truth = [t.strip() for t in a.truth.split(",")] if a.truth else ["?"] * len(a.images)

ck = torch.load(a.model, map_location="cpu")
C, S = ck["classes"], ck["img_size"]
m = timm.create_model("resnet18", pretrained=False, num_classes=len(C)); m.load_state_dict(ck["state_dict"]); m.eval()
tf = T.Compose([T.Resize(int(S * 256 / 224)), T.CenterCrop(S), T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

rows, n = [], len(a.images)
cols = min(n, 3); r = math.ceil(n / cols)
fig, axes = plt.subplots(r, cols, figsize=(5 * cols, 5.6 * r), squeeze=False)
for k, (path, gt) in enumerate(zip(a.images, truth)):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    t = time.perf_counter()
    with torch.no_grad():
        p = torch.softmax(m(tf(img).unsqueeze(0)), 1)[0]
    ms = (time.perf_counter() - t) * 1000
    v, i = torch.topk(p, 3)
    top = [(C[j], float(x)) for x, j in zip(v, i)]
    ok = None if (gt == "?" or gt not in C) else (top[0][0] == gt)
    rows.append(dict(image=path, truth=gt, top3=top, correct=ok, ms=ms))
    ax = axes[k // cols][k % cols]; ax.imshow(img); ax.axis("off")
    color = "darkorange" if ok is None else ("green" if ok else "red")
    txt = "\n".join(f"{c.replace('_', ' ')}: {s:.0%}" for c, s in top)
    ax.set_title(f"Gerçek: {gt.replace('_', ' ')}\n{txt}", color=color, fontsize=11, loc="left")
for k in range(n, r * cols):
    axes[k // cols][k % cols].axis("off")
plt.tight_layout(); plt.savefig("outputs/my_photos_results.png", dpi=130)
json.dump(rows, open("outputs/my_photos_results.json", "w"), indent=2, ensure_ascii=False)
for row in rows:
    print(row["image"], "| gerçek:", row["truth"], "| tahmin:", row["top3"][0], "| doğru:", row["correct"])

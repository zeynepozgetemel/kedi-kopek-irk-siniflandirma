"""Ek deney: aynı ırkın farklı çekim koşullarındaki fotoğrafları. Çıktı: outputs/experiment_same_breed.png/.json"""
import json, timm, torch
from PIL import Image, ImageOps
from torchvision import transforms as T
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ck = torch.load("outputs/best_model.pt", map_location="cpu"); C = ck["classes"]
m = timm.create_model("resnet18", num_classes=len(C)); m.load_state_dict(ck["state_dict"]); m.eval()
tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
rows = [("british_shorthair", [("IMG_6621", "İdeal: yüz net, yakın"), ("IMG_0676", "Çok yakın, el yüzde"),
                               ("IMG_0987", "Yandan/arkadan, uzak"), ("IMG_0984", "Yukarıdan, yüz görünmüyor")]),
        ("persian", [("IMG_7152", "İdeal: oturuyor, tam boy"), ("IMG_0153", "Kucakta, alttan, kol önünde")])]
fig, ax = plt.subplots(2, 4, figsize=(18, 11)); res = []
for r, (gt, items) in enumerate(rows):
    ax[r][0].text(-0.08, 0.5, gt.replace("_", " ").title(), transform=ax[r][0].transAxes, rotation=90,
                  va="center", ha="right", fontsize=14, weight="bold")
    for c in range(4):
        a = ax[r][c]; a.axis("off")
        if c >= len(items): continue
        f, cond = items[c]; im = ImageOps.exif_transpose(Image.open(f"my_photos/{f}.jpg")).convert("RGB")
        with torch.no_grad(): p = torch.softmax(m(tf(im)[None]), 1)[0]
        v, i = p.topk(3); top = [(C[j], float(x)) for x, j in zip(v, i)]; ok = top[0][0] == gt
        res.append(dict(image=f, condition=cond, truth=gt, top3=top, correct=ok, true_class_prob=float(p[C.index(gt)])))
        a.imshow(im); a.set_title(f"{cond}\n" + "\n".join(f"{n.replace('_', ' ')}: {s:.0%}" for n, s in top),
                                  color="green" if ok else "red", loc="left", fontsize=12)
plt.suptitle("Aynı ırk, farklı çekim koşulları (yeşil: doğru, kırmızı: yanlış)", fontsize=16); plt.tight_layout()
plt.savefig("outputs/experiment_same_breed.png", dpi=110)
json.dump(res, open("outputs/experiment_same_breed.json", "w"), indent=2, ensure_ascii=False)

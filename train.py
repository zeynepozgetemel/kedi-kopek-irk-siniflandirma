"""Oxford-IIIT Pet (37 ırk) - ResNet-18 transfer learning ile görüntü sınıflandırma.

Kullanım:  python train.py --data_dir data/data --epochs 8
Çıktılar:  outputs/ klasörüne model, metrikler ve grafikler.
"""
import argparse, io, json, os, random, time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, top_k_accuracy_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default="data/data")
p.add_argument("--out", default="outputs")
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--lr_backbone", type=float, default=1e-4)
p.add_argument("--lr_head", type=float, default=1e-3)
p.add_argument("--weight_decay", type=float, default=1e-4)
p.add_argument("--img_size", type=int, default=224)
p.add_argument("--workers", type=int, default=2)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Veri ----------------
def load(split):
    t = pq.read_table(os.path.join(args.data_dir, f"{split}-00000-of-00001.parquet"))
    meta = json.loads(t.schema.metadata[b"huggingface"])
    names = meta["info"]["features"]["label"]["names"]
    imgs = [d["bytes"] for d in t.column("image").to_pylist()]
    return imgs, t.column("label").to_pylist(), names

tr_imgs, tr_lbl, CLASSES = load("train")
te_imgs, te_lbl, _ = load("test")
# Resmi train(trainval) kümesinden %15 doğrulama ayrılır (sınıf dengeli), resmi test kümesi test olarak kalır.
idx_tr, idx_va = train_test_split(np.arange(len(tr_lbl)), test_size=0.15,
                                  stratify=tr_lbl, random_state=args.seed)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
train_tf = T.Compose([T.RandomResizedCrop(args.img_size, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                      T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize(int(args.img_size * 256 / 224)), T.CenterCrop(args.img_size),
                     T.ToTensor(), T.Normalize(MEAN, STD)])

class PetDS(Dataset):
    def __init__(self, imgs, labels, idx, tf):
        self.imgs, self.labels, self.idx, self.tf = imgs, labels, list(idx), tf
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return self.tf(Image.open(io.BytesIO(self.imgs[j])).convert("RGB")), self.labels[j]

dl_tr = DataLoader(PetDS(tr_imgs, tr_lbl, idx_tr, train_tf), args.batch_size, shuffle=True,
                   num_workers=args.workers, drop_last=True)
dl_va = DataLoader(PetDS(tr_imgs, tr_lbl, idx_va, eval_tf), 64, num_workers=args.workers)
dl_te = DataLoader(PetDS(te_imgs, te_lbl, range(len(te_lbl)), eval_tf), 64, num_workers=args.workers)
print(f"train={len(idx_tr)} val={len(idx_va)} test={len(te_lbl)} classes={len(CLASSES)} device={device}", flush=True)

# ---------------- Model ----------------
model = timm.create_model("resnet18.tv_in1k", pretrained=True, num_classes=len(CLASSES)).to(device)
head = list(model.get_classifier().parameters())
head_ids = {id(q) for q in head}
backbone = [q for q in model.parameters() if id(q) not in head_ids]
opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr_backbone},
                         {"params": head, "lr": args.lr_head}], weight_decay=args.weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr_backbone, args.lr_head],
                                            total_steps=args.epochs * len(dl_tr), pct_start=0.15)
crit = nn.CrossEntropyLoss(label_smoothing=0.1)

@torch.no_grad()
def predict(dl):
    model.eval(); P, Y, L = [], [], 0.0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        o = model(x); L += crit(o, y).item() * len(y)
        P.append(torch.softmax(o, 1).cpu()); Y.append(y.cpu())
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    return P, Y, L / len(Y)

# ---------------- Eğitim ----------------
hist, best, t0 = [], 0.0, time.time()
for ep in range(1, args.epochs + 1):
    model.train(); te = time.time(); tl, tc, n = 0.0, 0, 0
    for x, y in dl_tr:
        x, y = x.to(device), y.to(device)
        o = model(x); loss = crit(o, y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tl += loss.item() * len(y); tc += (o.argmax(1) == y).sum().item(); n += len(y)
    P, Y, vl = predict(dl_va); va = (P.argmax(1) == Y).mean()
    hist.append(dict(epoch=ep, train_loss=tl / n, train_acc=tc / n, val_loss=vl, val_acc=float(va),
                     epoch_sec=time.time() - te))
    print(json.dumps(hist[-1]), flush=True)
    if va > best:
        best = va
        torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "img_size": args.img_size,
                    "arch": "resnet18"}, os.path.join(args.out, "best_model.pt"))
train_time = time.time() - t0

# ---------------- Test ----------------
ck = torch.load(os.path.join(args.out, "best_model.pt"), map_location=device)
model.load_state_dict(ck["state_dict"])
P, Y, test_loss = predict(dl_te); pred = P.argmax(1)
pr, rc, f1, _ = precision_recall_fscore_support(Y, pred, average="macro", zero_division=0)
# Tahmin süresi: tek görüntü (batch=1), ön ısıtmadan sonra 50 ölçüm
model.eval(); x1 = torch.randn(1, 3, args.img_size, args.img_size).to(device)
with torch.no_grad():
    for _ in range(5): model(x1)
    ts = []
    for _ in range(50):
        s = time.perf_counter(); model(x1)
        if device == "cuda": torch.cuda.synchronize()
        ts.append((time.perf_counter() - s) * 1000)
metrics = dict(test_accuracy=float(accuracy_score(Y, pred)), test_top5_accuracy=float(top_k_accuracy_score(Y, P, k=5)),
               test_precision_macro=float(pr), test_recall_macro=float(rc), test_f1_macro=float(f1),
               test_loss=float(test_loss), best_val_accuracy=float(best),
               train_time_min=train_time / 60, inference_ms_per_image=float(np.median(ts)),
               device=device, torch=torch.__version__, cpu_threads=torch.get_num_threads(),
               n_train=len(idx_tr), n_val=len(idx_va), n_test=len(Y), args=vars(args), history=hist)
json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
rep = classification_report(Y, pred, target_names=CLASSES, digits=3, zero_division=0)
open(os.path.join(args.out, "classification_report.txt"), "w").write(rep)
print(rep); print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))

# ---------------- Grafikler ----------------
e = [h["epoch"] for h in hist]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(e, [h["train_loss"] for h in hist], "o-", label="train"); ax[0].plot(e, [h["val_loss"] for h in hist], "o-", label="val")
ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(e, [h["train_acc"] for h in hist], "o-", label="train"); ax[1].plot(e, [h["val_acc"] for h in hist], "o-", label="val")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(args.out, "training_curves.png"), dpi=150); plt.close()
cm = confusion_matrix(Y, pred, normalize="true")
plt.figure(figsize=(12, 10)); plt.imshow(cm, cmap="Blues", vmin=0, vmax=1); plt.colorbar()
plt.xticks(range(len(CLASSES)), CLASSES, rotation=90, fontsize=7); plt.yticks(range(len(CLASSES)), CLASSES, fontsize=7)
plt.xlabel("Tahmin"); plt.ylabel("Gerçek"); plt.title("Confusion Matrix (test, satır normalize)")
plt.tight_layout(); plt.savefig(os.path.join(args.out, "confusion_matrix.png"), dpi=150); plt.close()

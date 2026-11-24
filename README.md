# Kedi & Köpek Irkı Sınıflandırıcı (ResNet-18 · Oxford-IIIT Pet)

**Zeynep Özge Temel - 23120205072**

Bilgisayarlı görü ödevi. Model Google Colab'da GPU ile eğitildi (8 epoch, 3,4 dk, test doğruluğu %90,5): 37 kedi/köpek ırkını sınıflandıran, transfer learning ile eğitilmiş bir ResNet-18 modeli ve Gradio arayüzü.

## Kurulum
```bash
pip install -r requirements.txt
python download_data.py          # veri seti -> data/data/
```

## Eğitilmiş model
`best_model.pt` (45 MB) GitHub'ın dosya sınırını aştığı için **Releases** bölümünde yer alır.
İndirip `outputs/` klasörüne koyun ya da `python train.py` ile yeniden eğitin.

## Eğitim ve değerlendirme
```bash
python train.py --data_dir data/data --epochs 8
```
Çıktılar `outputs/` klasörüne yazılır: `best_model.pt`, `metrics.json`, `classification_report.txt`, `training_curves.png`, `confusion_matrix.png`.

## Uygulama (Gradio)
```bash
python app.py                    # http://127.0.0.1:7860
```
Bir fotoğraf yükleyin, model en olası 5 ırkı olasılıklarıyla gösterir.

## Kendi fotoğraflarınızla test
```bash
python predict_photos.py my_photos/*.jpg --truth "british_shorthair,bombay,persian,pug,?"
```
Sonuç görseli: `outputs/my_photos_results.png`.

## Dosyalar
| Dosya | Açıklama |
|---|---|
| `train.py` | Veri bölme, eğitim, test metrikleri, grafikler |
| `app.py` | Gradio web uygulaması |
| `predict_photos.py` | Telefon fotoğraflarını toplu test eder |
| `download_data.py` | Veri setini indirir |
| `experiment_same_breed.py` | Aynı ırkın farklı çekim koşullarındaki testi |
| `pet_breed_classifier.ipynb` | Tümünü Colab'da çalıştırmak için defter |

Veri seti: Oxford-IIIT Pet — https://www.robots.ox.ac.uk/~vgg/data/pets/

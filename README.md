> **EN:** A comparative study analyzing the impact of four outlier treatment methods (None, IQR Removal, Winsorization, Z-Score) on Linear Regression model performance using a food delivery dataset.

---

# Gıda Teslimat Süresi Tahmini: Aykırı Değer Yöntemlerinin Model Performansına Etkisi

Bu çalışmanın amacı, farklı aykırı değer işleme yöntemlerinin Doğrusal Regresyon modelinin istatistiksel performansına ve residual sağlığına nasıl etki ettiğini karşılaştırmalı olarak göstermektir.

---

## 🛠️ Kullanılan Araçlar

Python — Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

---

## 📊 Veri Seti

Kaggle üzerinden temin edilen **Food Delivery Time Estimation** veri seti kullanılmıştır. 500 sipariş kaydı içeren veri setinde şu değişkenler yer almaktadır:

| Değişken | Açıklama |
|---|---|
| `distance_km` | Teslimat mesafesi |
| `rider_speed` | Kurye hızı |
| `weather` | Hava durumu (Clear, Rainy, Snowy, Stormy, Windy) |
| `delivery_time` | Teslimat süresi — hedef değişken |

> Veri seti eğitim amaçlı kullanılmıştır. Proje tamamen öğrenme ve analiz pratiği amacıyla yapılmıştır.

---

## ⚙️ Metodoloji

### 1. Keşifsel Veri Analizi
- Mesafe ile teslimat süresi arasındaki ilişki saçılım grafiğiyle incelendi
- Korelasyon matrisiyle değişkenler arası ilişki ölçüldü (distance_km: 0.75, rider_speed: -0.45)
- Aykırı değerler kutu grafikleriyle tespit edildi; `delivery_time` değişkeninde sağ kuyrukta birkaç aykırı gözlem belirlendi

### 2. Ön İşleme
- `order_id` sütunu kaldırıldı
- `weather` kategorik değişkenine One-Hot Encoding uygulandı
- `distance_km` ve `rider_speed` StandardScaler ile standardize edildi
- Data leakage önlemek için scaler yalnızca train setine fit edildi

### 3. Aykırı Değer Yöntemleri
Dört farklı yaklaşım yalnızca eğitim setindeki `delivery_time` değişkenine uygulanarak karşılaştırıldı:

| Yöntem | Açıklama | Veri Kaybı |
|---|---|---|
| Base Model | Aykırı değer işlemi yapılmadı | — |
| IQR Removal | 1.5×IQR sınırları dışındaki satırlar silindi | 7 satır |
| Winsorization | %5–%95 dışındaki değerler sınıra çekildi | 0 satır |
| Z-Score Removal | \|z\| > 3 olan satırlar silindi | 5 satır |

### 4. Model Eğitimi
Her veri seti için aynı pipeline uygulandı: %80 train / %20 test, `random_state=42`. Test seti tüm modeller için sabit tutuldu.

---

## 📈 Sonuçlar

### Model Karşılaştırması

| Model | R² | RMSE | Base'e Göre Fark |
|---|---|---|---|
| **Base Model** | **0.8709** | **7.62 dk** | — |
| IQR Removal | 0.8695 | 7.66 dk | RMSE ↑ %0.5 |
| Winsorization | 0.8666 | 7.75 dk | RMSE ↑ %1.6 |
| Z-Score Removal | 0.8702 | 7.64 dk | RMSE ↑ %0.3 |

> **Not:** Hiçbir aykırı değer yöntemi Base Model'i geçemedi. En iyi performans, herhangi bir müdahale yapılmayan orijinal veri setiyle elde edildi.

### Residual Analizi

| Model | Residual Ortalaması | Std Dev | Yorum |
|---|---|---|---|
| Base Model | -0.48 | 7.65 | En düşük std dev — en tutarlı |
| IQR Removal | +0.13 | 7.70 | Ortalama 0'a yakın ama std arttı |
| Winsorization | +0.11 | 7.79 | En yüksek std dev |
| Z-Score Removal | -0.07 | 7.68 | Ortalama 0'a en yakın |

---

## 🔑 Ana Çıkarımlar

1. **Base Model en iyi test performansını verdi.** Tüm aykırı değer yöntemleri RMSE'yi hafifçe kötüleştirdi. Bu bulgu, aykırı değer temizliğinin her durumda performansı artırmadığını somut olarak göstermektedir.

2. **Aykırı değerler gerçek dünya varyasyonunu temsil edebilir.** Teslimat sürelerindeki yüksek değerler (örn. fırtınalı havalardaki gecikmeler) gerçek operasyonel senaryolara karşılık gelebilir. Bu değerleri silmek veya sınırlamak, modelin bu durumları öğrenmesini engelleyerek genelleme kapasitesini düşürmüş olabilir.

3. **Z-Score yöntemi en az zarar veren alternatif oldu.** En az veri kaybıyla (5 satır) ve en düşük residual ortalamasıyla (-0.07) Base Model'e en yakın sonucu üretti.


---

## 🗂️ Proje Dosyaları

| Dosya | Açıklama |
|---|---|
| `food_delivery_regression.py` | Tüm analizin kodu |
| `01_scatter_distance_vs_time.png` | Ham ilişki saçılım grafiği |
| `02_boxplots_raw.png` | Ham veri kutu grafikleri |
| `03_correlation_matrix.png` | Korelasyon matrisi |
| `04_boxplots_outlier_comparison.png` | 4 yöntem yan yana kutu grafikleri |
| `05–08_fit_*.png` | Her model için gerçek vs tahmin grafikleri |
| `09–12_residuals_*.png` | Her model için residual analizleri |

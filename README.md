> > **EN:** A comparative study analyzing the impact of four outlier treatment methods (None, IQR Removal, Winsorization, Z-Score) on Linear Regression model performance using a food delivery dataset.

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
- Aykırı değerler kutu grafikleriyle tespit edildi

### 2. Ön İşleme
- `order_id` sütunu kaldırıldı
- `weather` kategorik değişkenine One-Hot Encoding uygulandı
- `distance_km` ve `rider_speed` StandardScaler ile standardize edildi
- Data leakage önlemek için scaler yalnızca train setine fit edildi

### 3. Aykırı Değer Yöntemleri
Dört farklı yaklaşım `delivery_time` değişkenine uygulanarak karşılaştırıldı:

| Yöntem | Açıklama | Veri Kaybı |
|---|---|---|
| Base Model | Aykırı değer işlemi yapılmadı | — |
| IQR Removal | 1.5×IQR sınırları dışındaki satırlar silindi | 8 satır |
| Winsorization | %5–%95 dışındaki değerler sınıra çekildi | 0 satır |
| Z-Score Removal | \|z\| > 3 olan satırlar silindi | 5 satır |

### 4. Model Eğitimi
Her veri seti için aynı pipeline uygulandı: %80 train / %20 test, `random_state=42`.

---

## 📈 Sonuçlar

### Model Karşılaştırması

| Model | R² | RMSE | İyileşme |
|---|---|---|---|
| Base Model | 0.8709 | 7.62 dk | — |
| IQR Removal | 0.8833 | 6.52 dk | RMSE ↓ %14.5 |
| Winsorization | 0.8661 | 7.31 dk | RMSE ↓ %4.1 |
| Z-Score Removal | **0.8934** | 6.80 dk | RMSE ↓ %10.8 |

### Residual Analizi

| Model | Residual Ortalaması | Yorum |
|---|---|---|
| Winsorization | +0.20 | 0'a en yakın — en dengeli |
| Z-Score Removal | -0.34 | İkinci en dengeli |
| Base Model | -0.48 | Kabul edilebilir |
| IQR Removal | -0.99 | Sistematik negatif sapma |

---

## 🔑 Ana Çıkarımlar

1. **En iyi metrik** Z-Score yönteminde elde edildi (R²: 0.89)
2. **En iyi RMSE** IQR yönteminde elde edildi (%14.5 iyileşme)
3. **Winsorization** veri kaybetmeden iyileştirme sağladı ancak en düşük etki burada görüldü
4. **Residual analizi** önemli bir ödünleşim ortaya koydu: aykırı değer temizliği metrikleri iyileştirirken residual dağılımında sistematik sapmalara yol açtı. Bu durum, aykırı değerlerin her zaman gürültü olmadığını; bazen gerçek dünya varyasyonunu temsil edebileceğini göstermektedir

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

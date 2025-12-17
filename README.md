[README.md](https://github.com/user-attachments/files/24223735/README.md)
# 📈 BIST Hisse Senedi Trend Tahmin & AutoML Sistemi

Bu proje, **Borsa İstanbul (BIST)** verilerini kullanarak teknik analiz indikatörleri ve makine öğrenmesi (AutoML) modelleri ile hisse senedi trendlerini tahmin eden kapsamlı bir sistemdir.

Sistem, ham veriyi **İş Yatırım** servislerinden çeker, gelişmiş indikatörler (FINH, KAMA, ZLMA vb.) ile işler, **PyCaret** kullanarak en iyi ML modelini eğitir ve sonuçları interaktif bir **Web Raporu** (HTML) olarak sunar.

## 🚀 Özellikler

- **Otomatik Veri Toplama:** Son 5 yıllık hisse verilerini otomatik çeker.
- **Gelişmiş Feature Engineering:** Pine Script indikatörlerinin Python portları (FINH, KAMA, BlueLine, HHLL, OVT, LRB, ZLMA).
- **AutoML Entegrasyonu:** `PyCaret` ile birden fazla modelin (Random Forest, XGBoost, CatBoost vb.) otomatik kıyaslanması ve optimize edilmesi.
- **Sızıntı (Leakage) Koruması:** Geleceği bilen özellikleri eğitimden çıkararak gerçekçi test sonuçları.
- **İnteraktif Görselleştirme:** Al/Sat sinyallerini, model güven skorlarını ve kârlılık durumlarını gösteren HTML tabanlı dashboard.

## 📂 Dosya Yapısı ve İşleyiş

Proje 4 ana aşamadan oluşur:

### 1. Veri Toplama (`isyat_veri.py`)
*   `isyatirimhisse` kütüphanesini kullanarak belirlenen hisse senetlerinin (yaklaşık 100+ hisse) son 5 yıllık OHLCV (Açılış, Yüksek, Düşük, Kapanış, Hacim) verilerini çeker.
*   **Çıktı:** `hisse_verileri_2y.xlsx`

### 2. Özellik Mühendisliği (`generate_ml_features.py`)
*   Ham veriyi işleyerek ML modeli için anlamlı öznitelikler (features) üretir.
*   **İndikatörler:**
    *   **FINH:** Özelleştirilmiş EMA tabanlı trend takipçisi.
    *   **KAMA:** Kaufman Adaptive Moving Average.
    *   **HHLL:** Higher Highs / Lower Lows (Trend Yönü).
    *   **ZLMA:** Zero Lag Moving Average.
    *   **Diğerleri:** OVT, LRB, BlueLine.
*   **Etiketleme (Labeling):** 7 farklı indikatörün ortak kararına göre "Mevcut Trend" (0 veya 1) belirlenir ve hedef değişken (`TARGET_3D`) 3 gün sonrasına ötelenerek oluşturulur.
*   **Çıktı:** `ml_filtre_verileri.xlsx`

### 3. Model Eğitimi (`autoML.py`)
*   Hazırlanan veri seti üzerinde **PyCaret** kullanarak sınıflandırma modelleri eğitir.
*   `_PriceAbove` gibi hedef değişkenle doğrudan ilişkili (sızıntı yaratabilecek) kolonları eğitimden çıkarır.
*   Modelleri karşılaştırır, en iyisini seçer (örn. Extra Trees, Random Forest) ve hiperparametre optimizasyonu yapar.
*   Feature Importance ve Confusion Matrix grafiklerini kaydeder.
*   **Çıktı:** `.pkl` uzantılı model dosyası (örn. `fintech_best_model.pkl`).

### 4. Raporlama ve Görselleştirme (`visualize_signals_web.py`)
*   Eğitilen modelleri (V2, V3 vb.) yükler ve güncel veriler üzerinde tahmin yapar.
*   Tahmin güven skoru (Confidence Score) belirli bir eşiğin (örn. %55) üzerindeyse sinyalleri dikkate alır.
*   Sonuçları Plotly ve Jinja2 kullanarak tek bir HTML dosyasında toplar.
*   **Çıktı:** `Sinyal_Analiz_Raporu.html`

## 🛠 Kurulum

Proje Python 3.11+ sürümü ile uyumludur. Gerekli kütüphaneleri kurmak için:

```bash
pip install pandas pycaret openpyxl isyatirimhisse plotly jinja2 numpy
```

*Not: `pycaret` kurulumu bazen sistem bağımlılıkları gerektirebilir.*

## 💻 Kullanım Adımları

Sistemi sıfırdan çalıştırmak için aşağıdaki adımları sırasıyla uygulayın:

1.  **Verileri Güncelle:**
    ```bash
    python isyat_veri.py
    ```
    *(Bu işlem internet hızına bağlı olarak birkaç dakika sürebilir)*

2.  **Özellikleri (Features) Oluştur:**
    ```bash
    python generate_ml_features.py
    ```

3.  **Modeli Eğit (Opsiyonel - Eğer yeni model lazımsa):**
    ```bash
    python autoML.py
    ```

4.  **Raporu Oluştur:**
    ```bash
    python visualize_signals_web.py
    ```

5.  **Sonucu İncele:**
    Oluşan `Sinyal_Analiz_Raporu.html` dosyasını tarayıcınızda açın.

## 📊 Rapor İçeriği

HTML raporu iki sekmeden oluşur:
1.  **Grafik Analizi:** Seçilen hisse üzerinde model tahminlerini (AL/SAT) ve gerçek fiyat hareketlerini mum grafiği üzerinde gösterir.
2.  **Sinyal Özeti:** Tüm hisseler için son üretilen sinyalin tarihi, türü ve o sinyalden bu yana oluşan potansiyel kâr/zarar durumunu tablo halinde sunar.

## ⚠️ Yasal Uyarı

Bu proje **eğitim ve araştırma amaçlıdır**. İçerdiği sinyaller ve analizler **Yatırım Tavsiyesi Değildir (YTD)**. Finansal piyasalar yüksek risk içerir; modeller geçmiş verilere dayanır ve geleceği garanti edemez.

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
import sys

# generate_ml_features dosyasından hesaplama fonksiyonunu ve ayarları alıyoruz
from generate_ml_features import calculate_all_filters, CONFIG

warnings.filterwarnings('ignore')

# ============================================ 
# AYARLAR
# ============================================ 
FEATURE_CONFIG = {
    'input_file': 'hisse_verileri_2y.xlsx',  # Güncel veri dosyanız
    'output_file': f'280_gunluk_feature_seti_.xlsx', # Çıktı dosyası
    'days_to_keep': 280 # Son kaç günün verisi tutulacak?
}

def main():
    print("=" * 60)
    print(f"GÜNLÜK TAHMİN İÇİN SON {FEATURE_CONFIG['days_to_keep']} GÜNLÜK FEATURE OLUŞTURUCU")
    print("=" * 60)
    
    # 1. Veri Kontrolü
    if not os.path.exists(FEATURE_CONFIG['input_file']):
        print(f"❌ Hata: Girdi dosyası bulunamadı: {FEATURE_CONFIG['input_file']}")
        return

    print(f"📂 Veri okunuyor: {FEATURE_CONFIG['input_file']}")
    df = pd.read_excel(FEATURE_CONFIG['input_file'])
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    print(f"📊 Toplam {len(df)} satır, {df['CODE'].nunique()} hisse senedi verisi yüklendi.")
    
    # 2. Feature Hesaplama
    print("\n🔄 İndikatörler ve özellikler hesaplanıyor...")
    
    all_last_rows = []
    stocks = sorted(df['CODE'].unique())
    total_stocks = len(stocks)
    
    for idx, stock in enumerate(stocks, 1):
        # İlerleme göstergesi
        if idx % 10 == 0 or idx == total_stocks:
            print(f"\r   İşleniyor: [{idx}/{total_stocks}] {stock}", end="")
            
        stock_df = df[df['CODE'] == stock].copy().sort_values('DATE').reset_index(drop=True)
        
        # Yeterli veri kontrolü (Warm-up süresi için)
        if len(stock_df) < CONFIG['warmup_bars'] + FEATURE_CONFIG['days_to_keep'] + 5:
            continue
            
        try:
            # is_inference=True ile çağırıyoruz:
            # 1. Target hesaplanmaz (Geleceği bilmiyoruz)
            # 2. Son satırlar silinmez (Bugünün verisi bize lazım)
            features_df = calculate_all_filters(stock_df, CONFIG, is_inference=True)
            
            # Son N günü alıyoruz (Trend değişimi takibi için)
            last_rows = features_df.tail(FEATURE_CONFIG['days_to_keep']).copy()
            
            if not last_rows.empty:
                all_last_rows.append(last_rows)
                
        except Exception as e:
            # Hata olsa bile devam et, diğer hisseleri etkilemesin
            continue
            
    print("\n✅ Hesaplama tamamlandı.")
    
    if not all_last_rows:
        print("❌ Hiçbir hisse için özellik üretilemedi!")
        return
        
    # 3. Birleştirme ve Kaydetme
    final_df = pd.concat(all_last_rows, ignore_index=True)
    
    # Kategorik verileri string'e çevirelim (Excel'de daha temiz görünür)
    cat_cols = ['HHLL_Trend', 'HHLL_Trend_Lag1', 'HHLL_Trend_Lag2', 'HHLL_Trend_Lag3']
    for col in cat_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].astype(str)

    # Çıktı dosyasını kaydet
    final_df.to_excel(FEATURE_CONFIG['output_file'], index=False)
    
    print("\n" + "=" * 60)
    print(f"💾 Feature seti kaydedildi: {FEATURE_CONFIG['output_file']}")
    print(f"📊 Toplam {len(final_df)} satır veri (Her hisse için son {FEATURE_CONFIG['days_to_keep']} gün) hazırlandı.")
    print("🚀 Bu dosyayı modelinize 'predict' işlemi için verebilirsiniz.")
    print("=" * 60)

if __name__ == "__main__":
    main()

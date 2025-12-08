import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# KONFİGÜRASYON
# ============================================
CONFIG = {
    # Dosya Ayarları
    'input_file': 'hisse_verileri_2y.xlsx',
    'output_file': 'ml_filtre_verileri.xlsx',
    
    # Warm-up Süresi (Filtrelerin stabilizasyonu için gereken minimum bar sayısı)
    # Bu sayıdan sonraki veriler çıktıya dahil edilir
    'warmup_bars': 300,  # Manuel olarak ayarlanabilir
    
    # FINH Parametreleri
    'finh_period': 110,
    
    # KAMA Parametreleri
    'kama_period': 21,
    
    # BlueLine Parametreleri
    'blueline_period': 144,
    
    # HHLL Parametreleri
    'hhll_left_bars': 3,
    'hhll_right_bars': 3,
    
    # OVT Parametreleri
    'ovt_period': 89,
    
    # LRB Parametreleri
    'lrb_period': 105,
    
    # ZLMA Parametreleri
    'zlma_period': 144,
    'zlma_smooth': 1,
    
    # Lag (Gecikme) Parametreleri
    'lag_days': [1, 2, 3]  # Kaç gün geriye gidilecek
}

# ============================================
# YARDIMCI FONKSİYONLAR
# ============================================

def calculate_wma(series, period):
    """Weighted Moving Average - Pine Script ta.wma() ile aynı"""
    weights = np.arange(1, period + 1)
    
    def wma_calc(x):
        if len(x) < period:
            return np.nan
        return np.sum(weights * x) / weights.sum()
    
    return series.rolling(window=period).apply(wma_calc, raw=True)

def calculate_ema_custom(series, period):
    """Custom EMA - FINH için özel EMA hesaplaması"""
    alpha = 2 / (period + 1)
    result = pd.Series(index=series.index, dtype=float)
    
    for i in range(len(series)):
        if i == 0 or pd.isna(result.iloc[i-1]):
            result.iloc[i] = series.iloc[i]
        else:
            result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
    
    return result

def calculate_finh(df, period):
    """FINH göstergesi - Pine Script ile birebir aynı"""
    close = df['CLOSING_TL']
    sqrt_period = np.sqrt(period)
    
    # İlk EMA'lar
    ema1 = calculate_ema_custom(close, period)
    ema2 = calculate_ema_custom(close, period / 2)
    
    # Raw FINH
    finh_raw = 2 * ema2 - ema1
    
    # Son EMA
    finh = calculate_ema_custom(finh_raw, sqrt_period)
    
    return finh

def calculate_kama(df, length):
    """KAMA göstergesi - Pine Script ile birebir aynı"""
    close = df['CLOSING_TL']
    
    # Noise hesaplama
    xvnoise = close.diff().abs()
    
    # Signal ve Noise
    nsignal = (close - close.shift(length)).abs()
    nnoise = xvnoise.rolling(window=length).sum()
    
    # Efficiency Ratio
    nefratio = nsignal / nnoise
    nefratio = nefratio.fillna(0)
    nefratio = nefratio.replace([np.inf, -np.inf], 0)
    
    # Smoothing constants
    nfastend = 0.666
    nslowend = 0.0645
    
    nsmooth = ((nefratio * (nfastend - nslowend)) + nslowend) ** 2
    
    # KAMA hesaplama
    kama = pd.Series(index=close.index, dtype=float)
    kama.iloc[0] = close.iloc[0]
    
    for i in range(1, len(close)):
        kama.iloc[i] = kama.iloc[i-1] + nsmooth.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
    
    return kama

def calculate_blueline(df, period):
    """BlueLine göstergesi - Pine Script ta.ema() ile aynı"""
    close = df['CLOSING_TL']
    
    ema1 = close.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    
    blueline = 3 * (ema1 - ema2) + ema3
    
    return blueline

def calculate_ovt(df, period):
    """OVT (Orta Vade Trend) - Pine Script WMA mantığı ile"""
    close = df['CLOSING_TL']
    
    # n2ma ve nma hesaplama
    n2ma = 2 * calculate_wma(close, int(np.round(period / 2)))
    nma = calculate_wma(close, period)
    diff = n2ma - nma
    
    # sqn hesaplama
    sqn = int(np.round(np.sqrt(period)))
    
    # Final WMA
    n1_ovt = calculate_wma(diff, sqn)
    
    return n1_ovt

def calculate_lrb(df, period):
    """LRB (Linear Regression) - Pine Script ta.linreg() ile aynı"""
    close = df['CLOSING_TL']
    
    def linreg_value(y):
        if len(y) < period or np.isnan(y).any():
            return np.nan
        x = np.arange(len(y))
        # Linear regression: y = mx + b
        coeffs = np.polyfit(x, y, 1)
        # Son değeri döndür (offset = 0)
        return coeffs[0] * (len(y) - 1) + coeffs[1]
    
    lrb = close.rolling(window=period).apply(linreg_value, raw=True)
    
    return lrb

def calculate_zlma(df, period, smooth):
    """ZLMA (Zero Lag Moving Average) - Pine Script ile birebir aynı"""
    close = df['CLOSING_TL']
    
    # İlk WMA
    wma1 = calculate_wma(close, period)
    
    # İkinci WMA (smooth ile)
    priceMA_zlma = calculate_wma(wma1, smooth)
    
    # Final ZLMA
    zlma = 2 * priceMA_zlma - calculate_wma(priceMA_zlma, period)
    
    return zlma

def detect_hhll_trend(df, left_bars, right_bars):
    """
    HHLL Trend Detection - Pine Script mantığına uygun
    Higher Highs/Higher Lows vs Lower Highs/Lower Lows
    Returns: 1 for uptrend, 0 for downtrend
    """
    high = df['HIGH_TL'].values
    low = df['LOW_TL'].values
    close = df['CLOSING_TL'].values
    
    n = len(df)
    trend = np.zeros(n)
    
    # Pivot high ve low tespiti
    pivot_highs = []
    pivot_lows = []
    
    for i in range(left_bars, n - right_bars):
        # Pivot High kontrolü
        is_pivot_high = True
        for j in range(i - left_bars, i):
            if high[j] >= high[i]:
                is_pivot_high = False
                break
        for j in range(i + 1, i + right_bars + 1):
            if high[j] > high[i]:
                is_pivot_high = False
                break
        
        if is_pivot_high:
            pivot_highs.append((i, high[i]))
        
        # Pivot Low kontrolü
        is_pivot_low = True
        for j in range(i - left_bars, i):
            if low[j] <= low[i]:
                is_pivot_low = False
                break
        for j in range(i + 1, i + right_bars + 1):
            if low[j] < low[i]:
                is_pivot_low = False
                break
        
        if is_pivot_low:
            pivot_lows.append((i, low[i]))
    
    # Resistance ve support seviyeleri
    resistance = np.full(n, np.nan)
    support = np.full(n, np.nan)
    
    for i, val in pivot_highs:
        resistance[i:] = val
    
    for i, val in pivot_lows:
        support[i:] = val
    
    # Forward fill
    resistance = pd.Series(resistance).fillna(method='ffill').values
    support = pd.Series(support).fillna(method='ffill').values
    
    # Trend belirleme: 1 = uptrend, 0 = downtrend
    for i in range(n):
        if not np.isnan(resistance[i]) and close[i] > resistance[i]:
            trend[i] = 1
        elif not np.isnan(support[i]) and close[i] < support[i]:
            trend[i] = 0
        elif i > 0:
            trend[i] = trend[i-1]
    
    return pd.Series(trend, index=df.index)

def calculate_label(df):
    """
    Label hesaplama - State Machine mantığı
    
    Label 1 olması için (ALIM koşulları - hepsi birlikte):
    - fiyat > FINH
    - fiyat > KAMA
    - fiyat > BlueLine
    - fiyat > LRB
    - OVT eğimi pozitif (1)
    - ZLMA eğimi pozitif (1)
    - HHLL uptrend (1)
    
    Label 0 olması için (SATIM koşulları - hepsi birlikte):
    - fiyat < FINH
    - fiyat < KAMA
    - fiyat < BlueLine
    - fiyat < LRB
    - OVT eğimi negatif (0)
    - ZLMA eğimi negatif (0)
    - HHLL downtrend (0)
    """
    label = pd.Series(0, index=df.index, dtype=int)
    
    # Başlangıç durumu: 0
    current_label = 0
    
    for i in range(len(df)):
        # ALIM koşulları kontrolü
        buy_conditions = (
            df['FINH_PriceAbove'].iloc[i] == 1 and
            df['KAMA_PriceAbove'].iloc[i] == 1 and
            df['BlueLine_PriceAbove'].iloc[i] == 1 and
            df['LRB_PriceAbove'].iloc[i] == 1 and
            df['OVT_Slope'].iloc[i] == 1 and
            df['ZLMA_Slope'].iloc[i] == 1 and
            df['HHLL_Trend'].iloc[i] == 1
        )
        
        # SATIM koşulları kontrolü
        sell_conditions = (
            df['FINH_PriceAbove'].iloc[i] == 0 and
            df['KAMA_PriceAbove'].iloc[i] == 0 and
            df['BlueLine_PriceAbove'].iloc[i] == 0 and
            df['LRB_PriceAbove'].iloc[i] == 0 and
            df['OVT_Slope'].iloc[i] == 0 and
            df['ZLMA_Slope'].iloc[i] == 0 and
            df['HHLL_Trend'].iloc[i] == 0
        )
        
        # State machine mantığı
        if buy_conditions:
            current_label = 1
        elif sell_conditions:
            current_label = 0
        # Eğer hiçbir koşul sağlanmıyorsa mevcut durumu koru
        
        label.iloc[i] = current_label
    
    return label

# ============================================
# FİLTRE HESAPLAMA
# ============================================

def calculate_all_filters(df, config):
    """Tüm filtreleri hesapla ve eğimlerini ekle"""
    
    df = df.copy()
    df = df.sort_values('DATE').reset_index(drop=True)
    
    # --- D. Hacim Normalizasyonu (Relative Volume) ---
    # Son 10 günün ortalama hacmine oranı
    # Eğer o günkü hacim ortalamanın 2 katıysa 2.0, yarısıysa 0.5 olur.
    vol_ma = df['VOLUME_TL'].rolling(window=10).mean()
    df['VOL_Rel'] = df['VOLUME_TL'] / vol_ma
    
    # ----------------------------------------------------------------
    # GÖSTERGELERİN HESAPLANMASI
    # ----------------------------------------------------------------
    
    indicators = ['FINH', 'KAMA', 'BlueLine', 'OVT', 'LRB', 'ZLMA']
    
    # 1. Temel İndikatör Hesaplamaları
    print("   - İndikatörler hesaplanıyor...", end=" ")
    df['FINH'] = calculate_finh(df, config['finh_period'])
    df['KAMA'] = calculate_kama(df, config['kama_period'])
    df['BlueLine'] = calculate_blueline(df, config['blueline_period'])
    df['OVT'] = calculate_ovt(df, config['ovt_period'])
    df['LRB'] = calculate_lrb(df, config['lrb_period'])
    df['ZLMA'] = calculate_zlma(df, config['zlma_period'], config['zlma_smooth'])
    df['HHLL_Trend'] = detect_hhll_trend(df, config['hhll_left_bars'], config['hhll_right_bars'])
    print("✓")
    
    # 2. Türetilmiş Özellikler (Dist, Slope, Above)
    print("   - Türetilmiş özellikler (Dist, Slope, Above) hesaplanıyor...", end=" ")
    for ind in indicators:
        # Binary Slope (Label için)
        df[f'{ind}_Slope'] = (df[f'{ind}'].diff() > 0).astype(int)
        
        # Binary Price Above (Label ve Feature için)
        df[f'{ind}_PriceAbove'] = (df['CLOSING_TL'] > df[f'{ind}']).astype(int)
        
        # Continuous Distance Pct (ML için)
        # (Fiyat - Filtre) / Filtre
        df[f'{ind}_Dist_Pct'] = (df['CLOSING_TL'] - df[f'{ind}']) / df[f'{ind}']
        
        # Continuous Slope Rate (ML için)
        # İndikatörün yüzdesel değişimi
        df[f'{ind}_Slope_Rate'] = df[f'{ind}'].pct_change()
    print("✓")
    
    # 3. Lag (Gecikme) Özellikleri
    # Dist_Pct, Slope_Rate ve HHLL_Trend için gecikmeli veriler
    print("   - Lag (gecikme) özellikleri hesaplanıyor...", end=" ")
    features_to_lag = ['HHLL_Trend']
    for ind in indicators:
        features_to_lag.append(f'{ind}_Dist_Pct')
        features_to_lag.append(f'{ind}_Slope_Rate')
        
    for lag in config['lag_days']:
        for col in features_to_lag:
            df[f'{col}_Lag{lag}'] = df[col].shift(lag)
    print("✓")

    # Label hesaplama - State Machine mantığı
    # NOT: Label hesaplarken yukarıdaki Binary (0/1) kolonları kullanır.
    print("   - Label ve Target hesaplanıyor...", end=" ")
    
    # 1. Mevcut Trend Durumu (Feature olarak kullanılacak)
    df['Current_Trend'] = calculate_label(df)
    
    # 2. Hedef Değişken (3 Gün sonraki trend ne olacak?)
    df['TARGET_3D'] = df['Current_Trend'].shift(-3)
    
    print("✓")
    
    # Warm-up periyodundan sonraki verileri al
    # Lag'ler oluştuğu için en büyük lag kadar ekstra veri atmamız gerekebilir ama
    # warmup_bars (300) zaten 3 günlük lag'i (3) fazlasıyla kapsıyor.
    warmup_bars = config['warmup_bars']
    
    # Hem warmup kısmını atıyoruz hem de son 3 satırı (Target NaN olduğu için) atıyoruz
    df_output = df.iloc[warmup_bars:-3].copy()
    
    # ÇIKTI KOLONLARI
    output_columns = [
        'CODE', 'DATE', 'CLOSING_TL', 'LOW_TL', 'HIGH_TL', 
        'VOL_Rel',  # Normalize edilmiş Hacim
    ]
    
    # İndikatör bazlı kolonları ekle
    for ind in indicators:
        # Temel Değer
        output_columns.append(ind)
        # Continuous Features
        output_columns.append(f'{ind}_Dist_Pct')
        output_columns.append(f'{ind}_Slope_Rate')
        # Binary Feature (İstek üzerine eklendi)
        output_columns.append(f'{ind}_PriceAbove')
        
        # Lag Features
        for lag in config['lag_days']:
            output_columns.append(f'{ind}_Dist_Pct_Lag{lag}')
            output_columns.append(f'{ind}_Slope_Rate_Lag{lag}')
            
    # HHLL ve Lagleri
    output_columns.append('HHLL_Trend')
    for lag in config['lag_days']:
        output_columns.append(f'HHLL_Trend_Lag{lag}')
        
    # Feature ve Target
    output_columns.append('Current_Trend')
    output_columns.append('TARGET_3D')
    
    # Sadece mevcut kolonları seç (Hata olmaması için kontrol)
    output_columns = [col for col in output_columns if col in df_output.columns]
    
    df_output = df_output[output_columns].reset_index(drop=True)
    
    return df_output

# ============================================
# ANA FONKSİYON
# ============================================

def main():
    print("=" * 60)
    print("MACHINE LEARNING VERİ SETİ OLUŞTURUCU")
    print("7 FİLTRE: FINH + KAMA + BlueLine + HHLL + OVT + LRB + ZLMA")
    print("=" * 60)
    
    # Konfigürasyon özeti
    print("\n📋 PARAMETRELİK AYARLAR:")
    print(f"   Warm-up Bars: {CONFIG['warmup_bars']}")
    print(f"   Lag Days: {CONFIG['lag_days']}")
    print(f"   FINH Period: {CONFIG['finh_period']}")
    print(f"   KAMA Period: {CONFIG['kama_period']}")
    print(f"   BlueLine Period: {CONFIG['blueline_period']}")
    print(f"   HHLL Left/Right Bars: {CONFIG['hhll_left_bars']}/{CONFIG['hhll_right_bars']}")
    print(f"   OVT Period: {CONFIG['ovt_period']}")
    print(f"   LRB Period: {CONFIG['lrb_period']}")
    print(f"   ZLMA Period/Smooth: {CONFIG['zlma_period']}/{CONFIG['zlma_smooth']}")
    
    # Veriyi oku
    print(f"\n📂 Dosya okunuyor: {CONFIG['input_file']}")
    df = pd.read_excel(CONFIG['input_file'])
    
    # Tarih formatını düzenle
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    print(f"✅ Toplam {len(df)} satır veri yüklendi")
    print(f"📊 Hisse sayısı: {df['CODE'].nunique()}")
    print(f"📅 Veri tarih aralığı: {df['DATE'].min().date()} - {df['DATE'].max().date()}")
    
    # Her hisse için filtre hesapla
    all_results = []
    
    stocks = sorted(df['CODE'].unique())
    total_stocks = len(stocks)
    
    for idx, stock in enumerate(stocks, 1):
        print(f"\n[{idx}/{total_stocks}] {stock} işleniyor...")
        
        stock_df = df[df['CODE'] == stock].copy().sort_values('DATE').reset_index(drop=True)
        
        # Minimum veri kontrolü
        min_required = CONFIG['warmup_bars'] + 50  # Warm-up + minimum işlenebilir veri
        
        if len(stock_df) < min_required:
            print(f"   ⚠ Yetersiz veri ({len(stock_df)} < {min_required})")
            continue
        
        try:
            # Filtreleri hesapla
            stock_output = calculate_all_filters(stock_df, CONFIG)
            
            print(f"   ✅ Tamamlandı ({len(stock_output)} satır çıktı)")
            
            all_results.append(stock_output)
            
        except Exception as e:
            print(f"   ❌ Hata: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Tüm sonuçları birleştir
    if len(all_results) > 0:
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Excel'e kaydet
        final_df.to_excel(CONFIG['output_file'], index=False)
        
        print("\n" + "=" * 60)
        print(f"✅ Veri seti oluşturuldu!")
        print(f"💾 Sonuçlar kaydedildi: {CONFIG['output_file']}")
        print("=" * 60)
        
        # Özet istatistikler
        print("\n📊 ÖZET İSTATİSTİKLER:")
        print(f"   Toplam Satır: {len(final_df):,}")
        print(f"   Hisse Sayısı: {final_df['CODE'].nunique()}")
        print(f"   Tarih Aralığı: {final_df['DATE'].min().date()} - {final_df['DATE'].max().date()}")
        print(f"\n   Kolon Sayısı: {len(final_df.columns)}")
        
        # Label dağılımı
        print(f"\n   📈 TARGET_3D (HEDEF) DAĞILIMI:")
        # TARGET_3D float olabilir (shift yüzünden), int'e çevirelim veya direkt sayalım
        label_counts = final_df['TARGET_3D'].value_counts().sort_index()
        for label_val, count in label_counts.items():
            pct = (count / len(final_df)) * 100
            label_name = "ALIM (Pozitif)" if label_val == 1 else "SATIM (Negatif)"
            print(f"      {label_name}: {count:,} satır (%{pct:.2f})")
        
        # Eksik veri kontrolü
        print(f"\n   ⚠️ EKSİK VERİ ANALİZİ:")
        missing = final_df.isnull().sum()
        missing_pct = (missing / len(final_df)) * 100
        has_missing = False
        for col in final_df.columns:
            if missing[col] > 0:
                print(f"      {col}: {missing[col]:,} satır (%{missing_pct[col]:.2f})")
                has_missing = True
        
        if not has_missing:
            print(f"      Hiç eksik veri yok! ✓")
        
        return final_df
    else:
        print("\n❌ Hiç veri işlenemedi!")
        return None

if __name__ == "__main__":
    results = main()
    print("\n✨ Program başarıyla tamamlandı!")
from isyatirimhisse import fetch_stock_data
import datetime
import pandas as pd

# Hisse listesi
symbols = ['MAVI', 'EKGYO', 'TRGYO', 'RYSAS', 'BIMAS', 'GARAN', 'ENKAI', 'KCHOL', 'TUPRS', 'ASELS', 'MPARK', 'THYAO', 'EREGL'
] 

# Tarih aralığı (son 5 yıl)
today = datetime.date.today()
start_date = (today - datetime.timedelta(days=7 * 365)).strftime('%d-%m-%Y')
end_date = today.strftime('%d-%m-%Y')

# Verileri toplayacak liste
dataframes = []

for symbol in symbols:
    print(f"{symbol} için veri çekiliyor...")
    try:
        df_raw = fetch_stock_data(
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date,
            save_to_excel=False
        )

        if df_raw is None or df_raw.empty:
            print(f"{symbol} için veri alınamadı.")
            continue

        # İlgili sütunları seç
        df_filtered = df_raw[[
            'HGDG_HS_KODU',
            'HGDG_TARIH',
            'HGDG_KAPANIS',
            'HGDG_MIN',
            'HGDG_MAX',
            'HGDG_HACIM'
        ]].copy()

        # Sütun adlarını değiştir
        df_filtered.rename(columns={
            'HGDG_HS_KODU': 'CODE',
            'HGDG_TARIH': 'DATE',
            'HGDG_KAPANIS': 'CLOSING_TL',
            'HGDG_MIN': 'LOW_TL',
            'HGDG_MAX': 'HIGH_TL',
            'HGDG_HACIM': 'VOLUME_TL'
        }, inplace=True)

        dataframes.append(df_filtered)

    except Exception as e:
        print(f"{symbol} için hata oluştu: {e}. Hisse atlanıyor.")

# Tüm verileri birleştir ve kaydet
if dataframes:
    df_final = pd.concat(dataframes, ignore_index=True)
    filename = "hisse_verileri_2y.xlsx"
    try:
        df_final.to_excel(filename, index=False)
        print(f"Veriler '{filename}' dosyasına başarıyla kaydedildi.")
    except Exception as e:
        print("Excel dosyasına yazarken hata oluştu:", e)
else:
    print("Hiçbir geçerli veri alınamadı.")

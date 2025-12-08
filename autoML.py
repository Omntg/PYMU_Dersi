import pandas as pd
import numpy as np
from pycaret.classification import *
import os

# ============================================ 
# AYARLAR
# ============================================ 
CONFIG = {
    'input_file': 'ml_filtre_verileri.xlsx',
    'target_col': 'TARGET_3D',
    # 'Current_Trend' çıkarıldı çünkü TARGET_3D ile çok yüksek korelasyonlu (Data Leakage/Persistence)
    'ignore_cols': ['CODE', 'DATE', 'Current_Trend'], 
    'train_size': 0.8,               
    'session_id': 123,
    'log_experiment': False,
    'experiment_name': 'fintech_trend_prediction'
}

def run_pycaret_automl():
    print("="*60)
    print("🚀 PYCARET AUTOML BAŞLATILIYOR (LEAKAGE FIX UYGULANDI)")
    print("="*60)

    # 1. Veriyi Oku
    print(f"\n📂 Veri okunuyor: {CONFIG['input_file']}")
    if not os.path.exists(CONFIG['input_file']):
        print(f"❌ HATA: Dosya bulunamadı! ({CONFIG['input_file']})")
        return

    df = pd.read_excel(CONFIG['input_file'])
    print(f"✅ Veri yüklendi. Boyut: {df.shape}")

    # ------------------------------------------------------------
    # LEAKAGE FIX 3: Formül Sızıntısını Önleme (Sadece PriceAbove)
    # ------------------------------------------------------------
    # Veri setinde binary '_Slope' kolonları bulunmuyor (sadece Slope_Rate var).
    # Ancak '_PriceAbove' (0/1) kolonları var ve bunlar Target formülünün bir parçası.
    # Modelin ezber yapmasını önlemek için bu binary kolonları çıkarıyoruz.
    
    leak_cols = [c for c in df.columns if c.endswith('_PriceAbove')]
    
    # Mevcut ignore listesine ekle
    current_ignore = set(CONFIG['ignore_cols'])
    current_ignore.update(leak_cols)
    CONFIG['ignore_cols'] = list(current_ignore)
    
    print(f"\n🚫 Sızıntı önlemi: {len(leak_cols)} adet '_PriceAbove' özelliği eğitimden çıkarıldı.")
    # ------------------------------------------------------------

    # Eksik verileri temizle
    df = df.dropna(subset=[CONFIG['target_col']])
    
    # Tarihe göre sırala
    if 'DATE' in df.columns:
        df = df.sort_values('DATE')
        print("✅ Veriler tarihe göre sıralandı.")

    # 2. PyCaret Setup
    print("\n⚙️ PyCaret Setup yapılıyor...")
    
    s = setup(
        data=df,
        target=CONFIG['target_col'],
        ignore_features=CONFIG['ignore_cols'],
        train_size=CONFIG['train_size'],
        data_split_shuffle=False,      
        data_split_stratify=False,
        fold_strategy='timeseries',    
        fold=3,                        
        session_id=CONFIG['session_id'],
        verbose=False,
        html=False,
        log_experiment=CONFIG['log_experiment'],
        experiment_name=CONFIG['experiment_name']
    )
    
    print("✅ Setup tamamlandı.")
    
    # 3. Modelleri Karşılaştır
    print("\n🏎️ Modeller karşılaştırılıyor...")
    best_models = compare_models(n_select=3, sort='F1', verbose=True)
    
    best_model = best_models[0]
    print(f"\n🏆 En İyi Model: {best_model}")

    # 4. Optimize Et
    print("\n🏋️ Model optimize ediliyor...")
    tuned_model = tune_model(best_model, optimize='F1', fold=3, verbose=False)
    
    # 5. Sonuçlar
    print("\n📊 Test Seti Performansı:")
    predict_model(tuned_model)
    
    # 6. Feature Importance
    print("\n🔍 Feature Importance Kaydediliyor...")
    try:
        plot_model(tuned_model, plot='feature', save=True)
        print("✅ Feature Importance.png")
        
        plot_model(tuned_model, plot='confusion_matrix', save=True)
        print("✅ Confusion Matrix.png")
        
        # ----------------------------------------------------------
        # TÜM FEATURE IMPORTANCE SKORLARINI DIŞARI AKTAR
        # ----------------------------------------------------------
        # Modelin kullandığı tüm özelliklerin skorlarını alıp CSV'ye kaydedelim.
        # Böylece grafikte çıkmayan Dist_Pct gibi özellikleri de görebiliriz.
        
        # Modelin kendisini al (Pipeline içinden)
        model_obj = tuned_model
        
        # Eğer pipeline ise asıl modeli çekmeye çalış
        if hasattr(model_obj, 'steps'):
            model_obj = model_obj.steps[-1][1]
            
        if hasattr(model_obj, 'feature_importances_'):
            # Özellik isimlerini al
            feature_names = get_config('X_train').columns
            importances = model_obj.feature_importances_
            
            fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            fi_df = fi_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
            
            # CSV'ye kaydet
            fi_df.to_csv('feature_importance_all.csv', index=False)
            print("✅ 'feature_importance_all.csv' olarak tüm skorlar kaydedildi.")
            
            # İlk 20'yi ekrana bas
            print("\n🏆 TOP 20 ÖZELLİKLER:")
            print(fi_df.head(20))
            
            # Dist_Pct'lerin durumunu özel olarak göster
            print("\n📉 DISTANCE (UZAKLIK) ÖZELLİKLERİNİN SIRALAMASI:")
            dist_features = fi_df[fi_df['Feature'].str.contains('Dist_Pct')]
            print(dist_features)
        else:
            print("⚠️ Bu model türü feature_importances_ özniteliğine sahip değil.")
            
        # ----------------------------------------------------------

    except Exception as e:
        print(f"⚠️ Feature Importance hatası: {e}")

    # 7. Kaydet
    final_model = finalize_model(tuned_model)
    save_model(final_model, 'fintech_best_model')
    print("✅ Model kaydedildi.")

if __name__ == "__main__":
    run_pycaret_automl()
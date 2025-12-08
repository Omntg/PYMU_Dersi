import pandas as pd
from autogluon.tabular import TabularPredictor
import os

def train_and_predict(train_file, prediction_file, model_name_suffix):
    save_path = f'ag_models_{model_name_suffix}'
    predictor = None

    # 1. Model daha once egitilmis mi kontrol et (Kurtarma Modu)
    if os.path.exists(save_path) and os.path.exists(os.path.join(save_path, "predictor.pkl")):
        print(f"--- {save_path} bulundu, mevcut model yukleniyor... ---")
        try:
            predictor = TabularPredictor.load(save_path)
            print(f"Model basariyla yuklendi: {model_name_suffix}")
        except Exception as e:
            print(f"Model yuklenirken hata olustu, yeniden egitilecek: {e}")
    
    # 2. Eger model yuklenemediyse sifirdan egit
    if predictor is None:
        print(f"--- Processing {train_file} (Training) ---")
        try:
            train_data = pd.read_excel(train_file)
        except Exception as e:
            print(f"Error reading {train_file}: {e}")
            return None

        label = 'TARGET'
        if label not in train_data.columns:
            print(f"Error: '{label}' column not found in {train_file}")
            return None

        predictor = TabularPredictor(label=label, path=save_path).fit(
            train_data, 
            presets='medium_quality',
            num_bag_folds=5, # OOF tahminleri uretebilmek icin bagging aciyoruz
            num_bag_sets=1,  # Hizli olmasi icin tek set
            num_stack_levels=0 # Stacking yapma (hiz kazandirir)
        )

    # Duzeltme: get_model_best() yerine model_best
    print(f"Best model for {model_name_suffix}: {predictor.model_best}")

    # 3. Tahmin Asamasi
    print(f"--- Predicting using {prediction_file} ---")
    try:
        predict_data = pd.read_excel(prediction_file)
    except Exception as e:
        print(f"Error reading {prediction_file}: {e}")
        return None
    
    # Target sutunu varsa cikar (tahmin dosyasinda olmamali)
    label = 'TARGET'
    if label in predict_data.columns:
        predict_data = predict_data.drop(columns=[label])
        
    predictions = predictor.predict(predict_data)
    
    results = predict_data.copy()
    results[f'PREDICTION_{model_name_suffix}'] = predictions
    
    output_file = f'predictions_{model_name_suffix}.xlsx'
    results.to_excel(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return results

def main():
    prediction_file = 'gunluk_feature_seti_20251125.xlsx'
    
    # 3 gunluk model (Zaten egitildi, sadece yukleyip tahmin yapacak)
    train_and_predict(
        train_file='ml_filtre_verileri_3_gun.xlsx',
        prediction_file=prediction_file,
        model_name_suffix='3_gun'
    )
    
    # 5 gunluk model (Henuz egitilmedi, sifirdan baslayacak)
    train_and_predict(
        train_file='ml_filtre_verileri_5_gun.xlsx',
        prediction_file=prediction_file,
        model_name_suffix='5_gun'
    )

if __name__ == "__main__":
    main()
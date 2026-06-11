import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from src.preprocessing import load_data, prepare_data, NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES

DATA_PATH   = "data/diabetes_prediction_dataset.csv"
REPORT_PATH = "monitoring/evidently_report.html"

def generate_report():
    df = load_data(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = prepare_data(df)

    cat_encoder   = preprocessor.named_transformers_['cat']
    cat_names     = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = NUMERIC_FEATURES + BINARY_FEATURES + cat_names

    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_test  = pd.DataFrame(X_test,  columns=feature_names)

    df_train['target'] = y_train.values
    df_test['target']  = y_test.values

    print(f"Reference (train) : {df_train.shape[0]} lignes")
    print(f"Current (test)    : {df_test.shape[0]} lignes")

    report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset(),
    ])

    snapshot = report.run(
        reference_data = df_train,
        current_data   = df_test
    )

    os.makedirs("monitoring", exist_ok=True)
    snapshot.save_html(REPORT_PATH)

    print(f"\nRapport sauvegarde : {REPORT_PATH}")
    print("Ouvrez monitoring/evidently_report.html dans votre navigateur.")

if __name__ == "__main__":
    generate_report()
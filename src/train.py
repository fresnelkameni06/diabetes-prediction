import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from src.preprocessing import load_data, prepare_data

DATA_PATH         = "data/diabetes_prediction_dataset.csv"
MODEL_PATH        = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

def train_classifiers(X_train, X_val, X_test, y_train, y_val, y_test):
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100, scale_pos_weight=10,
            random_state=42, eval_metric='logloss'),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "DecisionTree": DecisionTreeClassifier(
            class_weight='balanced', random_state=42)
    }

    best_auc   = 0
    best_model = None
    best_name  = ""

    mlflow.set_tracking_uri(f"file:///{os.path.abspath('mlruns')}")
    mlflow.set_experiment("diabetes-prediction")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):

            # Entrainement
            model.fit(X_train, y_train)

            # Metriques
            auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
            auc_val   = roc_auc_score(y_val,   model.predict_proba(X_val)[:, 1])
            auc_test  = roc_auc_score(y_test,  model.predict_proba(X_test)[:, 1])
            ecart     = round(auc_train - auc_test, 4)

            # Logging MLflow — metriques uniquement (pas le modele)
            mlflow.log_param("model",      name)
            mlflow.log_param("threshold",  0.1)
            mlflow.log_metric("auc_train", auc_train)
            mlflow.log_metric("auc_val",   auc_val)
            mlflow.log_metric("auc_test",  auc_test)
            mlflow.log_metric("ecart_auc", ecart)

            # Affichage
            print(f"\n{'='*50}")
            print(f"MODELE : {name}")
            print(f"{'='*50}")
            print(f"  AUC Train : {auc_train:.4f}")
            print(f"  AUC Val   : {auc_val:.4f}")
            print(f"  AUC Test  : {auc_test:.4f}")
            print(f"  Ecart     : {ecart:.4f} {'OVERFITTING' if ecart > 0.1 else 'OK'}")
            print(classification_report(
                y_test, model.predict(X_test), zero_division=0))

            if auc_test > best_auc:
                best_auc   = auc_test
                best_model = model
                best_name  = name

    print(f"\n{'='*50}")
    print(f"Meilleur modele auto : {best_name} (AUC={best_auc:.4f})")
    print(f"Modele sauvegarde    : XGBoost (force)")
    print(f"{'='*50}")

    # On force XGBoost comme modele de production
    return models["XGBoost"], "XGBoost"

def train():
    # Chargement
    df = load_data(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = prepare_data(df)

    # Entrainement + selection
    best_model, best_name = train_classifiers(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    # Sauvegarde modele + preprocessor
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    with open(PREPROCESSOR_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)

    print(f"\nFichiers sauvegardes :")
    print(f"  -> {MODEL_PATH}")
    print(f"  -> {PREPROCESSOR_PATH}")

if __name__ == "__main__":
    train()
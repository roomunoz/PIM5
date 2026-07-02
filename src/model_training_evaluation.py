

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier

from model_utils import summarize_classification

CONFIG_MODELOS = [
    ('Logistic Regression', LogisticRegression, {
        'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42
    }),
    ('Random Forest', RandomForestClassifier, {
        'n_estimators': 300, 'class_weight': 'balanced',
        'max_depth': 10, 'min_samples_leaf': 5, 'random_state': 42, 'n_jobs': -1
    }),
    ('XGBoost', XGBClassifier, {
        'n_estimators':300,
        'learning_rate':0.05,
        'max_depth':6,
        'eval_metric':'logloss',
        'random_state':42
    })
]

def build_model(model_class, X_train, y_train, **kwargs):
    model = model_class(**kwargs)
    model.fit(X_train, y_train)
    return model


def evaluar_modelo(model, X_test, y_test, nombre: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = summarize_classification(y_test, y_pred, y_prob, nombre)

    reporte = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        'Modelo':      nombre,
        'ROC-AUC':     round(roc_auc, 4),
        'Accuracy':    round(reporte['accuracy'], 4),
        'Precision-0': round(reporte['0']['precision'], 4),
        'Recall-0':    round(reporte['0']['recall'], 4),
        'F1-0':        round(reporte['0']['f1-score'], 4),
        'Precision-1': round(reporte['1']['precision'], 4),
        'Recall-1':    round(reporte['1']['recall'], 4),
        'F1-1':        round(reporte['1']['f1-score'], 4),
    }


def entrenar_y_evaluar(model_class, kwargs, nombre, X_train, y_train, X_test, y_test) -> tuple:
    if nombre == 'XGBoost':
        kwargs['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()

    model = build_model(model_class, X_train, y_train, **kwargs)
    metricas = evaluar_modelo(model, X_test, y_test, nombre)
    return model, metricas


def comparar_modelos(lista_metricas: list) -> pd.DataFrame:
    df_comp = pd.DataFrame(lista_metricas)
    df_comp = df_comp.sort_values('Recall-0', ascending=False).reset_index(drop=True)
    mejor = df_comp.iloc[0]['Modelo']

    print("\n" + "="*85)
    print(" TABLA COMPARATIVA DE MODELOS")
    print("="*85)
    print(df_comp.to_string(index=False))
    print(f"\n Mejor modelo por Recall-0 (deteccion de morosos): {mejor}")
    print("="*85)

    return df_comp


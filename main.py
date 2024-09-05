from sklearn.model_selection import train_test_split
from utils.evaluate_models import evaluate_models
from utils.models_and_params import *
from utils.preprocessing import *
from utils.plot_results import plot_results


def main():
    df = load_data("final_genotipos_GERAL_RISK.csv")

    X = df.drop(columns=["risk"])
    y = df["risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train, X_test = preprocess_data(X_train, X_test)
    selector_array = train_selectors(X_train, X_test, y_train, y_test)

    models_and_params = (
        lr_model_and_params(),
        svm_model_and_params(),
        knn_model_and_params(),
        rf_model_and_params(),
        xgb_model_and_params()
    )

    evaluate_models(X_train, X_test, y_train, y_test, X.columns, selector_array, models_and_params, tune=False)
    evaluate_models(X_train, X_test, y_train, y_test, X.columns, selector_array, models_and_params, tune=True)

    plot_results("results/train/standart", "roc_auc", "ROC AUC de Diferentes Modelos (Treino)", "Número de SNPs", "ROC AUC")
    plot_results("results/test/standart", "roc_auc", "ROC AUC de Diferentes Modelos (Teste)", "Número de SNPs", "ROC AUC")
    plot_results("results/train/tuned", "roc_auc", "ROC AUC de Diferentes Modelos (Treino - Ajustado)", "Número de SNPs", "ROC AUC")
    plot_results("results/test/tuned", "roc_auc", "ROC AUC de Diferentes Modelos (Teste - Ajustado)", "Número de SNPs", "ROC AUC")

if __name__ == "__main__":
    main()

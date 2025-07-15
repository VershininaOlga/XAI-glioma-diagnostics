import argparse
import pandas as pd
from pathlib import Path
from src.data_prep import data_prep
from src.classifier import classifier
from src.survival_predictor import survival_predictor


def parse_args():
    parser = argparse.ArgumentParser(description='Prediction of glioma subtype and assessment of 12-year survival')
    parser.add_argument('--file_name', default='data.xlsx', help='Data file name')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    X_raw = pd.read_excel(Path('data')/args.file_name, index_col='Patient ID')

    preparer = data_prep()
    X_raw = preparer.check(X_raw)
    X_norm = preparer.znorm(X_raw)

    clf = classifier()
    predictions = pd.DataFrame({'Patient ID': X_norm.index, 'Glioma subtype': clf.predict(X_norm)}, index=X_norm.index)
    predictions['Glioma subtype'] = predictions['Glioma subtype'].map({0: 'Astrocytoma', 1: 'Oligodendroglioma', 2: 'Glioblastoma'})
    clf.plot_local_shap(X_norm)
    
    surv_pred = survival_predictor()
    predictions['12-year survival probability'] = surv_pred.predict(X_norm)
    surv_pred.plot_local_shap(X_norm)

    pred_prob_fn = Path('results/predictions.xlsx')
    pred_prob_fn.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_excel(pred_prob_fn, index=False)


if __name__ == "__main__":
    main()
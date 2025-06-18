import shap
import pickle
import numpy as np
import torchtuples as tt
from pycox.models import CoxCC
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class survival_predictor:
    def __init__(self):
        with open('models/sp_info.pkl', 'rb') as file:
            self.model_info = pickle.load(file)
        self.features = self.model_info['features']
        self.hparams = self.model_info['hparams']
        self.bgrd = self.model_info['bgrd']
        self.seed = self.model_info['seed']

        self.net = tt.practical.MLPVanilla(in_features=self.hparams['in_features'], 
                                           out_features=self.hparams['out_features'],
                                           num_nodes=self.hparams['num_nodes'], 
                                           dropout=self.hparams['dropout'], 
                                           output_bias=self.hparams['output_bias'], 
                                           batch_norm=self.hparams['batch_norm'])
        self.predictior = CoxCC(self.net, tt.optim.Adam(lr=self.hparams['lr']))
        self.predictior.load_net(path='models/sp.pt')
    
    def predict(self, X_norm):
        sample_ids = X_norm.index.values
        X = np.array(X_norm.values, dtype='float32')
        if not hasattr(self.predictior, 'baseline_hazards_'):
            _ = self.predictior.compute_baseline_hazards(batch_size=self.hparams['batch_size'])
        survival_function = self.predictior.predict_surv_df(X, batch_size=self.hparams['batch_size'])
        survival_function = survival_function.T
        survival_function.index = sample_ids
        survival_t = survival_function[survival_function.columns.values[np.argmin(np.abs(survival_function.columns.values - 12*365))]].values
        return survival_t

    def plot_local_shap(self, X_norm, dpi=100):
        print('Constructing explainability plots for survival prediction model...')
        explainer = shap.Explainer(self.predict, self.bgrd, feature_names=self.features, seed=self.seed)
        explain_values = explainer(X_norm)

        for sample_id in range(0, len(X_norm)):
            fig, ax = plt.subplots(figsize=(8, 7))

            shap_values = explain_values[sample_id, :].values
            base_value = explain_values[sample_id, :].base_values
            data_values = explain_values[sample_id, :].data
            features = np.array(explain_values.feature_names)
        
            shap_values_abs = np.abs(shap_values)
            inds_sort = np.argsort(shap_values_abs)
            features_sort = features[inds_sort]
            shap_values_sort = shap_values[inds_sort]
            data_values_sort = data_values[inds_sort]
            
            x_start = base_value
            y_start = 0.0
            y_h = 0.05
            y_gap = 0.01
            text_width = 0.14
            ylim_add = 0.02
            
            ax.plot([base_value, base_value], [-ylim_add, 0], c='silver', ls='dotted')
        
            yticks = []
            yticklabels = []
            for i in range(0, len(shap_values_sort)):
                length = shap_values_sort[i]
                if length > 0:
                    color = 'crimson'
                    x_arrow = 0.01
                    text_start = x_start + length + 0.001
                    test_color = 'crimson'
                else:
                    color = 'dodgerblue'
                    x_arrow = -0.01
                    text_start = x_start + length - 0.135
                    test_color = 'dodgerblue'
        
                x_1, y_1 = x_start, y_start
                x_2, y_2 = x_start + length - x_arrow, y_start + y_h
                x_3, y_3 = x_start + length, y_start + y_h/2
        
                ax.axhline(y_3, c='whitesmoke', ls='dotted', zorder=0)
                
                yticks.append(y_3)
                yticklabels.append(f'{round(data_values_sort[i], 3)} = {features_sort[i]}')
        
                if abs(length) > abs(x_arrow):
                    vertices = [(x_1, y_1), (x_1, y_2), (x_2, y_2), (x_3, y_3), (x_2, y_1)]
                else:
                    vertices = [(x_1, y_1), (x_1, y_2), (x_3, y_3)]
                
                poly = Polygon(vertices, closed=True, facecolor=color)
                ax.add_patch(poly)
        
                test_in = f'+{length:.{3}f}' if length > 0 else f'{length:.{3}f}'
                ax.text(text_start, y_start + y_h/2.5, test_in, c=test_color, fontsize=14)
        
                ax.plot([x_3, x_3], [y_1, y_2+y_gap], c='silver', ls='dotted')
        
                x_start = x_start + length
                y_start = y_start + y_h + y_gap
        
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels, fontsize=16)
            ax.set_ylim(-ylim_add, y_2+ylim_add)
        
            ax.set_xlim(-0.14, 1.14)
            ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)
        
            ax.text(base_value, -0.095, f'E[f(X)] = {round(base_value, 3)}', c='dimgray', fontsize=16)
        
            plt.figtext(0.3, 0.00, 'Overall survival probability', fontsize=16)
        
            ax.text(x_3-0.1, y_2+ylim_add, f'f(x) = {round(x_3, 3)}', c='dimgray', fontsize=16)
            ax.plot([x_3, x_3], [-ylim_add, y_3], c='silver', ls='dotted', zorder=0)
        
            pict_fn = Path(f'results/survival_prediction_local_expl/{X_norm.index.values[sample_id]}_sp.png')
            pict_fn.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(pict_fn, dpi=dpi, bbox_inches='tight', facecolor='w')
            plt.close()

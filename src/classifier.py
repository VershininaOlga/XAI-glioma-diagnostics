import shap
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class classifier:
    def __init__(self):
        with open('models/clf.pkl', 'rb') as file:
            self.model = pickle.load(file)
        self.predictor = self.model['predictor']
        self.features = self.model['features']
        self.bgrd = self.model['bgrd']
        self.seed = self.model['seed']

    def predict(self, X_norm):
        y_pred = self.predictor.predict(X_norm)
        return y_pred
        
    def predict_proba(self, X_norm):
        y_pred_proba = self.predictor.predict_proba(X_norm)
        return y_pred_proba

    def plot_local_shap(self, X_norm, dpi=100):
        print('Constructing explainability plots for classification model...')
        explainer = shap.Explainer(self.predict_proba, self.bgrd, feature_names=self.features, seed=self.seed)
        explain_values = explainer(X_norm)

        for sample_id in range(0, len(X_norm)):
            fig, ax = plt.subplots(1, 3, figsize=(30, 7))
            for plot_class in [0, 1, 2]:
                shap_values = explain_values[sample_id, :, plot_class].values
                base_value = explain_values.base_values[:, plot_class][sample_id]
                data_values = explain_values[sample_id, :, plot_class].data
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
                
                ax[plot_class].plot([base_value, base_value], [-ylim_add, 0], c='silver', ls='dotted')
            
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
            
                    ax[plot_class].axhline(y_3, c='whitesmoke', ls='dotted', zorder=0)
                    
                    yticks.append(y_3)
                    yticklabels.append(f'{round(data_values_sort[i], 3)} = {features_sort[i]}')
            
                    if abs(length) > abs(x_arrow):
                        vertices = [(x_1, y_1), (x_1, y_2), (x_2, y_2), (x_3, y_3), (x_2, y_1)]
                    else:
                        vertices = [(x_1, y_1), (x_1, y_2), (x_3, y_3)]
                    
                    poly = Polygon(vertices, closed=True, facecolor=color)
                    ax[plot_class].add_patch(poly)
            
                    test_in = f'+{length:.{3}f}' if length > 0 else f'{length:.{3}f}'
                    ax[plot_class].text(text_start, y_start + y_h/2.5, test_in, c=test_color, fontsize=14)
            
                    ax[plot_class].plot([x_3, x_3], [y_1, y_2+y_gap], c='silver', ls='dotted')
            
                    x_start = x_start + length
                    y_start = y_start + y_h + y_gap
            
                ax[plot_class].spines['top'].set_visible(False)
                ax[plot_class].spines['right'].set_visible(False)
                ax[plot_class].spines['left'].set_visible(False)
            
                ax[plot_class].set_yticks(yticks)
                ax[plot_class].set_yticklabels(yticklabels, fontsize=16)
                ax[plot_class].set_ylim(-ylim_add, y_2+ylim_add)
            
                ax[plot_class].set_xlim(-0.14, 1.14)
                ax[plot_class].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax[plot_class].set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)
            
                ax[plot_class].text(base_value, -0.095, f'E[f(X)] = {round(base_value, 3)}', c='dimgray', fontsize=16)
            
                if plot_class == 0:
                    plt.figtext(0.155, 0.0, 'Probability of class "Astrocytoma"', fontsize=16)
                elif plot_class == 1:
                    plt.figtext(0.44, 0.00, 'Probability of class "Oligodendroglioma"', fontsize=16)
                elif plot_class == 2:
                    plt.figtext(0.73, 0.00, 'Probability of class "Glioblastoma"', fontsize=16)
            
                ax[plot_class].text(x_3-0.1, y_2+ylim_add, f'f(x) = {round(x_3, 3)}', c='dimgray', fontsize=16)
                ax[plot_class].plot([x_3, x_3], [-ylim_add, y_3], c='silver', ls='dotted', zorder=0)

            plt.subplots_adjust(wspace=0.4)
            pict_fn = Path(f'results/classification_local_expl/{X_norm.index.values[sample_id]}_clf.png')
            pict_fn.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(pict_fn, dpi=dpi, bbox_inches='tight', facecolor='w')
            plt.close()

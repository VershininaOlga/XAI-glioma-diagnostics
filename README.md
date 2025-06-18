# XAI-glioma-diagnostics

This repository contains XAI models for predicting glioma subtype (astrocytoma, oligodendroglioma or glioblastoma) and predicting 12-year survival probability of patients with glioma based on RNA-sequencing data (read counts - the number of mapped RNA-seq reads per gene) from 13 genes:
- TERT
- NOX4
- MMP9
- TRIM67
- ZDHHC18
- HDAC1
- TUBB6
- ADM
- NOG
- CHEK2
- KCNJ11
- KCNIP2
- VEGFA



## Getting Started

### File structure
```
├── data                          <- Folder containing data files
│   └── data.xlsx                     <- Data file example
│
├── images                         <- Folder containing examples of explainability plots
│
├── models                         <- Folder containing trained prediction models
│
├── results                       <- Results folder (will appear after the predictions are launched)
│
├── src                           <- Folder containing source code
│   ├── classifier.py                 <- Module for working with the model for classifying glioma subtypes
│   ├── normalizer.py                 <- Module for data normalization
│   └── survival_predictor.py         <- Module for working with the survival prediction model
│
├── requirements.txt              <- File for installing python dependencies
│
├── run_predictions.py             <- Script for running prediction models
│
└── README.md                     <- This file
```

### Requirements
Python 3.11  
openpyxl
pandas
matplotlib
scikit-learn==1.5.1
torch==2.0.0
torchtuples==0.2.2
pycox==0.3.0
shap==0.46.0
numpy==1.26.4 

### Installing
```bash
# clone project
git clone https://github.com/VershininaOlga/XAI-glioma-diagnostics.git
cd XAI-glioma-diagnostics

# [OPTIONAL] create environment
python -m venv .glioma
.glioma\Scripts\activate

# install requirements
pip install -r requirements.txt
```

### Data preparation
You need to prepare a data .xlsx file containing the following columns: Patient ID (unique patient identifier), TERT, NOX4, MMP9, TRIM67, ZDHHC18, HDAC1, TUBB6, ADM, NOG, CHEK2, KCNJ11, KCNIP2 and VEGFA.  
The file with your data should be placed in the ```data``` folder.  
An example data file can be found in ```data/data.xlsx```.  
Please note that the data should not contain missing values ​​(all samples with missing values ​​will be removed from consideration).

### Running the prediction model
To run the model:
```
python run_predictions.py --file_name <file_name>
```
where ```<file_name>``` - data file name, eg ```data.xlsx```

Please note that plotting explainability graphs for a large dataset can take a significant amount of time.

As a result of running the script, a ```results``` folder will be generated, which will contain  
- file with a table containing the predicted glioma subtype and 12-year survival probability for patients from data file
- subfolder ```classification_local_expl``` containing local explainability plots of the glioma subtype classification model for each patient
- subfolder ```survival_prediction_local_expl``` containing local explainability plots of the survival prediction model for each patient


### Interpretation of explainability plots for the glioma subtype classification model
![illustration](image/illustration_1.png)
For each patient, three graphs are constructed to explain the prediction of each possible class. For each class the y-axis displays the features (genes) from bottom to top in ascending order based on their contribution to the prediction. The x-axis shows the probability of predicting the corresponding class. The bottom of each graph shows the base probability of the model, E[f(X)], from which the prediction begins. Each band of the graph shows how much and in which direction the value of the feature changes the prediction. Blue bands correspond to features whose values ​​decrease the probability of the corresponding class, and red bands correspond to features whose values ​​increase the probability of the corresponding class. The resulting prediction of the model is the class with the highest probability f(x).


### Interpretation of explainability plots for the survival prediction model
![illustration](image/illustration_2.png)
The y-axis displays the features (genes) from bottom to top in ascending order based on their contribution to the prediction. The x-axis shows the 12-year ovarall survival probability. The bottom of the graph shows the base probability of the model, E[f(X)], from which the prediction begins. Each band of the graph shows how much and in which direction the value of the feature changes the prediction. Blue bands correspond to features whose values ​​decrease the survival probability, and red bands correspond to features whose values ​​increase the survival probability. The predicted probability of 12-year survival is reflected in f(x).


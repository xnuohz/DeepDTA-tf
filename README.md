# DeepDTA-tf
Tensorflow implementation of "DeepDTA deep drug-target binding affinity prediction"
## envs
* python 3.6
* tensorflow 1.12.0
## data
* kiba: 10 cross validation
* newkd: all for training
* FDA, Merck, XJ, YS, ZDC: just for prediction
## data format
* cross validation: ligands.csv, proteins.csv, inter.csv
* all training: ligands.csv, proteins.csv, pairs.csv
## benchmark
```txt
kiba as training set, predict new data(FDA & Merck) and calc AUC
```
## Config files
* data_xxx.cfg
```txt
[model]
data_path = path of training data
path = path for saving the model
```
* model.cfg
```txt
[model]
params...
```
* predict.cfg
```txt
Notice: change model.path & data.prediction for prediction
[model]
params...
path = ../../model/newkd

[data]
path = ../../data/
prediction = FDA,Merck,XJ,YS,ZDC
```
## How to run
* training
```bash
cd src/deepdta
# cross validation
python cv.py ../../config/model.cfg ../../config/data_kiba.cfg
# all for training
python main.py ../../config/model.cfg ../../config/data_newkd.cfg
```
* prediction
```bash
cd src/deepdta
python predict.py ../../config/predict.cfg
```
## Results
* AUCs

| dataset used in training | FDA | Merck | FDA + Merck
| :-: | :-: | :-: | :-: |
| kiba | 0.4803 | 0.5365 | 0.5083 |
| newkd | 0.5250 | 0.6264 | 0.5551 |
| DB201707 | 0.4921 | 0.6427 | 0.5979 |

## DrugBank 201707 Result
* AUC 5 cross validation

| type | 1 | 2 | 3 | 4 | 5 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| new target | 0.808 | 0.833 | 0.821 | 0.803 | 0.806 |
| new pair | 0.900 | 0.904 | 0.892 | 0.902 | 0.891 |
| new drug | 0.839 | 0.817 | 0.807 | 0.797 | 0.825 |

* AUPR 5 cross validation

| type | 1 | 2 | 3 | 4 | 5 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| new target | 0.447 | 0.515 | 0.503 | 0.340 | 0.372 |
| new pair | 0.622 | 0.609 | 0.578 | 0.634 | 0.584 |
| new drug | 0.467 | 0.444 | 0.396 | 0.403 | 0.387 |

## Drop P11388 col, other as training, P11388 as test
AUC: 0.581
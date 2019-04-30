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
Notice: change model.path & data.type & data.prediction for prediction
[model]
params...
path = ../../model/newkd

[data]
path = ../../data/
# affinity or classification
type = affinity
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
## AUC
| dataset used in training | FDA | Merck | FDA + Merck
| :-: | :-: | :-: | :-: |
| kiba | 0.4803 | 0.5365 | 0.5083 |
| newkd | 0.5250 | 0.6264 | 0.5551 |
| DB201707 | 0.4921 | 0.6427 | 0.5979 |
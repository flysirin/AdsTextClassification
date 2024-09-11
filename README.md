# Text Classification with PyTorch and Scikit-Learn

This project focuses on text classification using a neural network built with PyTorch. It leverages `CountVectorizer` from `scikit-learn` for text preprocessing and includes evaluation metrics such as accuracy, precision, recall, and F1-score to measure the model’s performance.
Contains [AppFlaskPyTorch](AppFlaskPyTorch) - server with Flask and PyTorch on docker, click for read more information.



## How to use
1. Put labeled data **excel** file in `./data/train_data` for train model   
Column with text - named - 'text'
Column with label - named - 'label'  

2. Run train model
 ```shell
   python train.py
   ```
3. Model and parameters save in `./saved/model/`

4. Put your data witch you want to classificate in `./data/for_prediction_data`
5. ```python run_predict_data.py```
6. Classificated data will be saved in `./data/output_data` with `label`

## Requirements
```shell
pip install -r requirements.txt
```
##### Contains
```
torch==2.4.0
pandas==2.2.2
scikit-learn==1.5.1
openpyxl
torchtext==0.18.0
numpy<2
```

`numpy<2 ` - downgrade version for exclude conflicts  

## Special requirements for using Cuda
```
pip install -r requirements_cuda.txt
```
##### Contains
```
# Specifying the source for PyTorch packages
--index-url https://download.pytorch.org/whl/cu121

# Packages with specific versions and source
torch==2.4.1
torchvision==0.19.1
torchaudio==2.4.1

# Packages from standard PyPI
--extra-index-url https://pypi.org/simple

pandas==2.1.2
scikit-learn==1.5.1
openpyxl
torchtext==0.18.0

```

#### Further development in progress


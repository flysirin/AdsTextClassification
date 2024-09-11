# Special application server with Flask. Used PyTorch for classification data


## How to use:

#### 1. Put in `./saved/models/`:  
`labels_encoded.json` - label indexes  
`model.pth` - current model  
`vectorizer.pkl` - model settings and other  

#### 2. Build Docker image
```shell
 docker build -t pytorch-prediction-service .
```
#### 3. Run container with comparing ports 5000:5000
```shell
docker run -d -p 5000:5000 --name my_pytorch_service pytorch-prediction-service
```
#### 4. Send post request on ```http://127.0.0.1:5000/predict``` with cleared data like:
`{id: 'some text', ...}` where id - key, 'some text' - text data for classification

#### 5. Server will answer `{id: 'label', ... }` , where 'label' - comparing number code from model and `labels_encoded.json`

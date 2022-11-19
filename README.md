# Language Classifier
Language classifier project to identify italian texts. Built on top of pytorch, pytorch-lightning for the model part, fastapi and onnxruntime for the api part. The APIs are ready to use with a trained model, but you can optionally train your own model using the built-in scripts.
## Description
The project is composed by:
- api
  - Manage the predict logic via API
  - Uses the onnxruntime for the inference
  - Use fastapi for the path and event loop logic
- model_code
  - Contains all the code necessary to train and test a new model
  - the preprocessing part
  - allow you to create your own vocabulary
  - contains an LSTM model implementation, TextDataset and all the pre-processing step (tokenization logic also)
- notebooks
  - Contain a notebook about data analysis
- train_report
  - Contains the export of the wandb dashboard of the trained model
- runtime
  - contains the onnx model and the vocabulary
- weights
  - contains the pytorch lightning checkpoint of the trained model

## Usage
### Preprocessing and split the data
To preprocess your dataset, go to the project root and run the preprocessor script:
`python model_code/data/preprocessing/preprocessor.py`
```bash
Usage: preprocessor.py [OPTIONS]

Options:
  -d, --dataset TEXT           [required]
  -v, --val-size FLOAT         [default: 0.3]
  -t, --test-size FLOAT        [default: 0.1]
  -T, --text-col TEXT          [default: Text]
  -y, --target-col TEXT        [default: Language]
  -s, --seed INTEGER           [default: 12]
  -o, --output-path TEXT       [required]
  --languages-to-exclude TEXT  [default: [], -e, --exclude]
  -l, --language TEXT          [default: Italian]
  --help                       Show this message and exit. 
```

The script will create 3 files (`train.csv`, `val.csv` and `test.csv`) inside the specified output-path.

### Create the vocabulary
To create the vocabulary for your data, go to the project root and run the preprocessor script:
`python model_code/data/preprocessing/create_vocabulary.py`

```bash
Usage: create_vocabulary.py [OPTIONS]

Options:
  -d, --dataset TEXT      [required]
  -t, --text-column TEXT  [default: Text]
  -f, --freq INTEGER      [default: 10]
  -o, --out TEXT          [required]
  --help                  Show this message and exit.
```

### Train the model
To train a new model, go to the project root and run the training script:
`python model_code/train.py`

```bash
Usage: train.py [OPTIONS]

Options:
  -b, --batch-size INTEGER  [default: 32]
  -t, --train TEXT          [default: model_code/data/datasets/train.csv]
  -v, --val TEXT            [default: model_code/data/datasets/val.csv]
  -T, --test TEXT           [default: model_code/data/datasets/test.csv]
  -v, --vocab TEXT          [default: model_code/data/vocabulary/vocab.pth]
  -d, --dropout FLOAT       [default: 0.2]
  -s, --emb-size INTEGER    [default: 256]
  -l, --lr FLOAT            [default: 0.02]
  -e, --epochs INTEGER      [default: 5]
  -g, --gpu
  --help                    Show this message and exit.
```
It will create N checkpoints inside the model folder.

### Test the model using a test-set 
To test a model, go to the project root and run the training script:
`python model_code/test.py`

```bash
Usage: test.py [OPTIONS]

Options:
  -m, --model TEXT  [required]
  -T, --test TEXT   [default: model_code/data/datasets/test.csv]
  -v, --vocab TEXT  [default: model_code/data/vocabulary/vocab.pth]
  -g, --gpu
  --help            Show this message and exit.
```


### Export the model (onnx)
To crete an onnx model you can use the `to_onnx_converter` script. Go to the model_code folder and run the script:
`python to_onnx_converter.py`

```bash
Usage: to_onnx_converter.py [OPTIONS]

Options:
  -m, --model TEXT  [required]
  -o, --out TEXT    [default: runtime]
  --help            Show this message and exit.
```
It will create an onnx model inside the output directory.


## Deploy
To deploy the inference APIs to use the model you can use docker-compose.
### Run the APIs (via docker-compose)
You can easly deploy the APIs using the command:
`docker-compose up` from the project root.
It will deploy an API server on `http://localhost:5000`
### Make inference
To make inference you can `POST` the api on `http://localhost:5000/predict` with a JSON body like:
```json
{
    "text":"Ciao, come stai?"
}
```
The API in this case will reply:
```json
{
    "prediction": 1,
    "class": "italian"
}
```

### Logging
The API log on stdout but also in two files `api_logging/info.txt` and `api_logging/error.txt`. The path is a bind of the container.


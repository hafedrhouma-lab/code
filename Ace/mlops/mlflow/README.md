### Run training
```bash
just jump-inside-mlflow-server
cd ./exmaple
python train.py --alpha 0.5 --l1-ratio 0.1
```

### Build base docker image
```bash
mlflow models build-docker --model-uri runs:/<RUN_ID>/model --name "<IMAGE_NAME>"
docker run -p 6001:8080 "<IMAGE_NAME>"
```

### Build custom docker image
```bash
mlflow models generate-dockerfile \
  --model-uri runs:/<RUN_ID>/model \
  --output-directory <DOCKERFILE_AND_MODEL_DIR> \
  --install-mlflow \
  --enable-mlserver
cd <DOCKERFILE_AND_MODEL_DIR>
docker build . -t <IMAGE_TAG> -f ./Dockerfile
docker run -p 6001:8080 <IMAGE_TAG>
```

### HTTP Access to the model inside container
```bash
curl http://127.0.0.1:6001/invocations -H 'Content-Type: application/json' -d '{
      "inputs": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
  }'
```

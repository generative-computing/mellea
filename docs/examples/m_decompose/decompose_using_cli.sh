#MODEL_ID=granite3.3:latest
#MODEL_ID=granite4:latest

MODEL_ID=mistral-small3.2:latest # granite4:latest

m decompose run --model-id $MODEL_ID  --out-dir ./ --input-file example.txt

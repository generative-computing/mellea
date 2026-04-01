#MODEL_ID=granite3.3
#MODEL_ID=granite4

MODEL_ID=mistral-small3.2 #qwen2.5:7b

m decompose run --model-id $MODEL_ID  --out-dir ./ --input-file example.txt


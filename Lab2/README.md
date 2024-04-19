# Lab2:  Butterfly & Moth Classification

## How to run
* To train your VGG19 model, run the following command
```bash
python main.py --model VGG19 --mode train --device cuda
```
* To run the script with specific parameters, use the following command
```bash
python main.py \
  --model ResNet50 \
  --mode train \
  --device cuda \
  --epoch 100 \
  --lr 1e-3
```
* To visualize the training process, run the following command then open the browser and go to `http://localhost:6006/`
```bash
tensorboard --logdir=runs
```
## Training Arguments
|Argument|Description|Default|
|---|---|---|   
|`'--model`|`VGG19`, `ResNet50`, `VGG19_pretrained`, `ResNet50_pretrained`|VGG19|
|`'--mode'`|`train`, `test`|train|
|`'--device'`|`cuda`, `cpu`|cuda|
|`'--epoch'`|Number of training epochs|400|
|`'--lr'`|Learning rate|1e-3|
## Test Results
make sure to save the model on checkpoint/{model_name}/best_val_{model_name}.pth
* test for your model
```bash
python main.py --model VGG19 --mode test
python main.py --model ResNet50 --mode test
python main.py --model VGG19_pretrained --mode test
python main.py --model ResNet50_pretrained --mode test
```
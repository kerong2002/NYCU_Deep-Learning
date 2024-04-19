# Lab1: Backpropagation

## Install dependencies
```bash
pip install numpy
pip install matplotlib
pip install tensorboardX
```
## How to run
* Run with default settings on linear problem
```bash
python main.py --problem linear
```
* To run the script with specific parameters, use the following command
```bash
python main.py \
  --problem XOR \
  --activation ReLU \
  --optimizer Adam \
  --epoch 10000 \
  --lr 1e-2 \
  --hidden 16 \
  --scheduler \
  --save_model \
  --tensorboard \
  --log_dir XOR_ReLU_Adam_10000_1e-2_16

```
* To visualize the training process, run the following command then open the browser and go to `http://localhost:6006/`
```bash
tensorboard --logdir=logs
```
## Training Arguments
|Argument|Description|Default|
|---|---|---|
|`'--problem'`|`linear`, `XOR`|linear|
|`'--activation'`|`sigmoid`, `ReLU`, `LeakyReLU`, `Tanh`, `Identity`|sigmoid|
|`'--optimizer'`|`SGD`, `SGDM`, `RMSProp`, `Adam`|SGD|
|`'--epoch'`|Number of training epochs|5000|
|`'--lr'`|Learning rate|5e-2|
|`'--hidden'`|Number of hidden layers|32|
|`'--scheduler'`|Use learning rate scheduler|false|
|`'--save_model'`|Save the trained model|false|
|`'--tensorboard'`|Use TensorBoard for visualization|false|
|`'--log_dir'`|Directory for saving logs|\tmp|

## Test Results
* test for linear problem
```bash
python main.py --problem linear --save_model
python test.py --problem linear
```
* test for XOR problem
```bash
python main.py --problem XOR --save_model
python test.py --problem XOR
```

# A PyTorch implementation of WSDDN
### Benchmarking
|**Learning rate**: 0.0001 (1e-4)|**LR Decay**: 10|**Optimizer**: SGD|
|-----------|------------|------------|

All results were obtained on *Pascal VOC 2007* (min scale = 600, **ROI Pool**):
|Backbone|Batch size|Max epoch|mAP|
|-------|-------|-------|-------|
|VGG16|1|6|30.9%|
|ResNet18|2|8|30.4%|
|ResNet50|2|6|30.6%|

---
## Preparation
Clone this repo and create `data` folder in it:
```
git clone https://github.com/akhilpm/WSDDN.git
cd WSDDN && mkdir data
```

### Prerequisites
- Python 3.5+
- PyTorch 1.3+
- CUDA 8+


**NOTE:** When using  PyTorch pretrained models, specify *RGB* color mode, image range = [0, 1], *mean = [0.485, 0.456, 0.406]* and *std = [0.229, 0.224, 0.225]* in additional parameters for run script. For example:
```
python run.py train ............. -ap color_mode=RGB image_range=1 mean="[0.485, 0.456, 0.406]" std="[0.229, 0.224, 0.225]"
```

### Data preparation
Prepare dataset as described [here](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) for Pascal VOC.
*Actually, you can use any dataset. Just download it and create softlinks in `library_root/data` folder.*

You can, *but not necessary*, specify directory name for dataset relative `./data` folder in addtional parameters for run script. For example:
- `python run.py train ............. --add_params devkit_path=VOC_DEVKIT_PATH` => ./data/VOC_DEVKIT_PATH
- `python run.py train ............. -ap data_path=COCO2014` => ./data/COCO2014

**NOTE:** Name of the parameter is different for datasets (`devkit_path` for Pascal VOC, `data_path` for COCO, etc.)

**WARNING! If you change any parameter of some dataset, you must remove cache files for this dataset in `./data/cache` folder!**

---
## Usage:
All interaction with the library is done through a `run.py` script. Just run:
```
python run.py -h
```
and follow help message.

### Train
To train WSDDN network with ResNet50 backbone on Pascal VOC 2012 trainval dataset in 10 epochs, run next:
```
python run.py train --net resnet 50 --dataset voc_2012_trainval --total-epoch 10 --cuda
```
Some parameters saved in [default config file](https://github.com/akhilpm/WSDDDN/blob/master/lib/config.py), another parameters has default values.

For more information about train script, you need to run `python run.py train -h` and follow help message.

### Test
If you want to evlauate the detection performance of above trained model, run next:
```
python run.py test --net resnet 50 --dataset voc_2012_test --epoch $EPOCH --cuda
```
where *$EPOCH* is early saved checkpoint epoch (maximum =10 for training example above).

For more information about test script, you need to run `python run.py test -h` and follow help message.

### Detect
If you want to run detection on your own images with above trained model:
* Put your images in `data/images` folder
* Run script:
```
python run.py detect --net resnet 50 --dataset voc_2012_test --epoch $EPOCH --cuda --vis
```
where *$EPOCH* is early saved checkpoint epoch (maximum =10 for training example above).

After detect, you will find the detection results in folder `dataT/images/result` folder.

For more information about detect script, you need to run `python run.py detect -h` and follow help message.

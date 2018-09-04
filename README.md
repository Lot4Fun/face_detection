# __Face Segmentation (In production)__

## Description

## Demo
Still very poor results.  

![Sample1](https://github.com/pystokes/face_segmentation/blob/master/docs/20020719_IMG00018.jpg)
![Sample2](https://github.com/pystokes/face_segmentation/blob/master/docs/20020725_IMG00438.jpg)
## Requirement

## Install
```
git clone https://github.com/pystokes/face_segmentation
```

## Usage
Modify [hparams.yaml](https://github.com/pystokes/face_segmentation/blob/master/hparams/hparams.yaml) before running following processes.

### Create dataset
```
python impulso.py data
```

### Prepare
```
python impulso.py prepare -d DATA-ID
```

### Train
To resume training, specify MODEL-ID.
```
python impulso.py train -e EXPERIMENT-ID [-m MODEL-ID]
```

### Test
```
python impulso.py test -e EXPERIMENT-ID -m MODEL-ID
```

### Predict
```
python impulso.py estimate -e EXPERIMENT-ID -m MODEL-ID -i DATA_DIR/DATA_FILE
```

## Contribution

## Licence
- Permitted: Private Use  
- Forbidden: Commercial Use  

## Author
[LotFun](https://github.com/pystokes)

## Specification
### Data to prepare by Aggregator.py
- IMPULSO_HOME: Absolute path to directory impulso.py exists

|Usage phase|Type|Path|
|:---|:---|:---|
|Train|Input|IMPULSO_HOME/datasets/{DATA-ID}/train/x/x.npy
|Train|Ground Truth|IMPULSO_HOME/datasets/{DATA-ID}/train/t/t.npy
|Test|Input|IMPULSO_HOME/datasets/test/x/x.npy
|Test|Ground Truth|IMPULSO_HOME/datasets/test/t/t.npy
|Test|Image file name|IMPULSO_HOME/datasets/test/x/filename.npy

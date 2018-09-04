# __Face Segmentation (In production)__

## Description

## Demo
Very poor result.  

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

### Predit
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
|Module|Class|Method|Input|Output|
|:---|:---|:---|:---|:---|
|impulso|Impulso|```__init__```|hparams_path|-|
|impulso|Impulso|dataset|-|datasets/{data_id}/x/x.npy <br> datasets/{data_id}/t/t.npy <br> datasets/test/x/x.npy <br> datasets/test/t/t.npy|
|impulso|Impulso|prepare|-|experiments/{experiment_id}/*|
|impulso|Impulso|train|-||
|impulso|Impulso|test|-||
|impulso|Impulso|estimate|-||




# __Face Segmentation (In production)__

## Description

## Demo
These images are the results at the moment.  
Not successfully detected by multiple people.  
Note: She is Minami HAMABE, a famous actress from Ishikawa prefecture in Japan.  

![Sample1](https://github.com/pystokes/face_segmentation/blob/master/docs/hamabe_minami_1.jpg)
![Sample2](https://github.com/pystokes/face_segmentation/blob/master/docs/hamabe_minami_2.jpg)
![Sample3](https://github.com/pystokes/face_segmentation/blob/master/docs/hamabe_minami_3.jpg)
![Sample3](https://github.com/pystokes/face_segmentation/blob/master/docs/hamabe_minami_4.jpg)
![Sample3](https://github.com/pystokes/face_segmentation/blob/master/docs/hamabe_minami_5.jpg)
![Sample3](https://github.com/pystokes/face_segmentation/blob/master/docs/hamabe_minami_6.jpg)

## Requirement

## Install
```
git clone https://github.com/pystokes/face_segmentation
```

## Usage
Modify [hparams.yaml](https://github.com/pystokes/face_segmentation/blob/master/hparams/hparams.yaml) before running following processes.

### Create dataset
```
python impulso.py dataset
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
python impulso.py estimate -e EXPERIMENT-ID -m MODEL-ID -x DATA_DIR -y OUTPUT_DIR
```

## Contribution

## Licence
- Permitted: Private Use  
- Forbidden: Commercial Use  

## Author
[Toshiyuki KITA](https://github.com/pystokes)

## Specification
### Data to be created with Aggregator.py
- IMPULSO_HOME: Absolute path to directory impulso.py exists

|Usage phase|Type|Path|
|:---|:---|:---|
|Train|Input|IMPULSO_HOME/datasets/{DATA-ID}/train/x/x.npy
|Train|Ground Truth|IMPULSO_HOME/datasets/{DATA-ID}/train/t/t.npy
|Test|Input|IMPULSO_HOME/datasets/test/x/x.npy
|Test|Ground Truth|IMPULSO_HOME/datasets/test/t/t.npy
|Test|Image file name|IMPULSO_HOME/datasets/test/x/filename.npy

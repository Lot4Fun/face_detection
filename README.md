# __Face Segmentation__

## Description
Detect faces and visualize the results as heatmap.

## Demo
These images are the latest results.  
Not successfully detected if there are mutiple or small faces.  

### 1st results
![Sample1](https://github.com/pystokes/face_segmentation/blob/master/docs/1st/hamabe_minami_1.jpg)
![Sample2](https://github.com/pystokes/face_segmentation/blob/master/docs/1st/hamabe_minami_2.jpg)
![Sample3](https://github.com/pystokes/face_segmentation/blob/master/docs/1st/hamabe_minami_3.jpg)
![Sample3](https://github.com/pystokes/face_segmentation/blob/master/docs/1st/hamabe_minami_4.jpg)
![Sample3](https://github.com/pystokes/face_segmentation/blob/master/docs/1st/hamabe_minami_5.jpg)
![Sample3](https://github.com/pystokes/face_segmentation/blob/master/docs/1st/hamabe_minami_6.jpg)

### Best results at the moment
![Sample1](https://github.com/pystokes/face_segmentation/blob/master/docs/best/hamabe_minami_1.jpg)
![Sample2](https://github.com/pystokes/face_segmentation/blob/master/docs/best/hamabe_minami_2.jpg)
![Sample3](https://github.com/pystokes/face_segmentation/blob/master/docs/best/hamabe_minami_3.jpg)
![Sample3](https://github.com/pystokes/face_segmentation/blob/master/docs/best/hamabe_minami_4.jpg)
![Sample3](https://github.com/pystokes/face_segmentation/blob/master/docs/best/hamabe_minami_5.jpg)
![Sample3](https://github.com/pystokes/face_segmentation/blob/master/docs/best/hamabe_minami_6.jpg)

## Requirement
tensorflow-gpu==1.4.0  
Keras==2.1.4  

## Install
```
git clone https://github.com/pystokes/face_segmentation
```

## Usage
### Modify below sources
1. Modify [Aggregator.py](https://github.com/pystokes/face_segmentation/blob/master/src/Aggregator.py) according to your data (See __Specification__ section below)

2. Create network structure like [ImpulsoNet.py](https://github.com/pystokes/face_segmentation/blob/master/src/model/ImpulsoNet.py) according to your data
3. Modify [Estimator.py](https://github.com/pystokes/face_segmentation/blob/master/src/Estimator.py) according to your data
4. Modify [Evaluator.py](https://github.com/pystokes/face_segmentation/blob/master/src/Evaluator.py) according to your data
5. Modify [hparams.yaml](https://github.com/pystokes/face_segmentation/blob/master/hparams/hparams.yaml) before running following processes.

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

## License
- Permitted: Private Use  
- Forbidden: Commercial Use  

## Author
[LotFun](https://github.com/pystokes)

## Specification
### Data to be created with [Aggregator.py](https://github.com/pystokes/face_segmentation/blob/master/src/Aggregator.py)
- IMPULSO_HOME: Absolute path to directory [impulso.py](https://github.com/pystokes/face_segmentation/blob/master/impulso.py) exists

|Usage phase|Type|Path|
|:---|:---|:---|
|Train|Input|IMPULSO_HOME/datasets/{DATA-ID}/train/x/x.npy
|Train|Ground Truth|IMPULSO_HOME/datasets/{DATA-ID}/train/t/t.npy
|Test|Input|IMPULSO_HOME/datasets/test/x/x.npy
|Test|Ground Truth|IMPULSO_HOME/datasets/test/t/t.npy
|Test|Image file name|IMPULSO_HOME/datasets/test/x/filename.npy

# __Face Segmentation__

## Description
Detect faces and visualize the results as heatmap.

## Demo
```
python impulso.py predict -e 0912-0121-1904 -m 70 -x ./tmp/input -y ./tmp/output
```

### Results
Not successfully detected if there are mutiple or small faces.  
![Sample1](https://github.com/pystokes/face_segmentation/blob/tmp/output/figures/hamabe_minami_1.jpg)
![Sample2](https://github.com/pystokes/face_segmentation/blob/tmp/output/figures/hamabe_minami_2.jpg)
![Sample3](https://github.com/pystokes/face_segmentation/blob/tmp/output/figures/hamabe_minami_3.jpg)
![Sample3](https://github.com/pystokes/face_segmentation/blob/tmp/output/figures/hamabe_minami_4.jpg)
![Sample3](https://github.com/pystokes/face_segmentation/blob/tmp/output/figures/hamabe_minami_5.jpg)
![Sample3](https://github.com/pystokes/face_segmentation/blob/tmp/output/figures/hamabe_minami_6.jpg)

## Requirement
tensorflow-gpu==1.4.0  
Keras==2.1.4  

## Install
```
git clone https://github.com/pystokes/face_segmentation
```

## Usage
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
python impulso.py estimate -e EXPERIMENT-ID -m MODEL-ID -x INPUT_DIR -y OUTPUT_DIR
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

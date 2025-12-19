# HDC for Computer Vision
### 

To run ConvHDC (V + B + SP) D = 10000:
```
python -m HDC_good_code.trainer_learnable -ds DATASET
```

To run ConvHDC (WB + B + SP) D = 10000:
```
python -m HDC_good_code.trainer_learnable -ds DATASET -wb true
```

To run ConvHDC (WB + B + MP) D = 10000:
```
python -m HDC_good_code.trainer_learnable -ds DATASET -wb true -mp true
```

To run ConvHDC  (WB + C + SP) D = 1000:
```
python -m HDC_good_code.trainer_learnable -ds DATASET -wb true -cm true -D 1000
```

To run ConvHDC (WB + C + MC) D = 1000:
```
python -m HDC_good_code.trainer_learnable -ds DATASET -wb true -cm true -mp true -D 1000
```

To run ResConvHDC (WB + C + SC) D = 1000:
```
python -m HDC_good_code.trainer_learnable -ds DATASET -wb true -cm true -D 1000 -ma ResConvHDC
```

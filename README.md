# Animation-to-Sketch-with-U-Net

## Team Members
We put our pictures into model to colorize and transform sketch for fun, not doing so well because we use animation picture to train instead of real pictures.

| 0816059 王行悌 |Style Change|
| -------- | -------- | 
|<img src="https://i.imgur.com/qtjerOl.jpg" width="200px">  |<img src="https://i.imgur.com/H3222Qq.jpg" width="200px">| 


| 0816125 張紀睿 |Convert to Sketch|
| -------- | -------- | 
|<img src="https://i.imgur.com/vFq4Lxw.jpg" width="200px">   |<img src="https://i.imgur.com/xTmP9EB.jpg" width="200px"> | 

## Introduction
Because we both like animation, we think that may be cool to turn animation into sketch. There are traditional ways to do that, but it's not very flexible. For example, XDoG is a method to recognize sketch, but you need to adjust parameters for each input picture, or you might a get nasty result. We try to resolve this problem, and build a robust model.

## Related work
- [U-Net: Convolutional Networks for Biomedical Image Segmentation 2015](https://arxiv.org/pdf/1505.04597.pdf)
    This paper proposed a new model on image-to-image problem doing segmentation. 
    
## Methodology: U-Net
We try to use U-Net to solve our sketch problem.
- Structure
  - Contracting path: left side structure extracts features.
  - Expanding path: right side structures upsamples features.
  ![](https://i.imgur.com/MnkTZZ9.png)
- Property
  - Global view: a layer's input not only come from last layer, but also previous layer. It's make the model know local and global view of the picture. It's the reason that makes to exactly position the object edge.
  - Free size input: U-Net use conv layer instead of fully connected layer, so it can deal any size of picture.

## Dataset
For our work, we need data with both origin pictures and sketches. In fact, it's hard to find picture with complete painting procedure, so we decide to get animation pictures first, and then create sketches by some traditional methods, such as XDoG (eXtended Difference-of-Gaussians).
- Animation picture
  We write a crawler script to get pictures on [konachan](https://konachan.com/post?tags=%20rating:safe) web. We download about 5000 pictures for training data.
- Sketch
  We write a XDoG script to turn animation pictures into sketches. Because of XDoG properties, if we want to get clear sketches, we need to adjust parameters with each pictures. The work is so heavy and nasty, so we just use same parameters in XDog, and choose clear sketches for training.

    | Input | XDoG sketch | Description |
    | -------- | -------- | -------- |
    | <img src="https://i.imgur.com/7S1ccVl.jpg" width="320px"> | <img src="https://i.imgur.com/9gXsa0r.jpg" width="320px"> | Clear |
    | <img src="https://i.imgur.com/ySECTLj.jpg" width="320px"> | <img src="https://i.imgur.com/mn7J8cV.jpg" width="320px"> | Not clear

## Experiments and Implement

The first model we trained was too deep, which made it hard to converge even if we raise the learning rate and use really small trainset.


| Input | 50 epoch | 500 epoch| 5000 epoch |
| -------- | -------- | -------- | -------- |
|![](https://i.imgur.com/dYDR5Xo.png)     | ![](https://i.imgur.com/0nbycP2.png)     | ![](https://i.imgur.com/SpIYFVZ.png)   | ![](https://i.imgur.com/LW9Qa5l.png)|

Therefore, we replaced the last DoubleConv layer by a normal Conv2d layer, and the result became much better. However, overfit occured so the model has poor performance on pictures that were not in trainset.

<img src="https://i.imgur.com/Bmvbb9a.png" width="250px">

Finally, we added dropout to prevent from overfitting, and we trained the model for 300 epoch with 210 pictures in the trainset. The loss function we use was L1Loss, and the optimizer was SGD with LR=1e-3, momentum=0.9. See the result in next session.

## Result
#### Training Loss

<img src="https://i.imgur.com/n4Jqa4c.png" width="350px">

#### Training


| Input | Ground Truth | Prediction |
| -------- | -------- | -------- |
| ![](https://i.imgur.com/org3Fww.jpg) |   ![](https://i.imgur.com/Yq2lilc.jpg)  | ![](https://i.imgur.com/GSeXV4J.png)     |
|![](https://i.imgur.com/bNp4aSN.jpg)    |  ![](https://i.imgur.com/bRPb9Go.jpg)   | ![](https://i.imgur.com/BX4j0Fn.png)|

#### Validation


| Input | Ground Truth | Prediction |
| -------- | -------- | -------- |
|![](https://i.imgur.com/FIqGQHX.jpg)  |![](https://i.imgur.com/kgCPkrW.jpg)   |![](https://i.imgur.com/M2Ibaho.png)  |
|![](https://i.imgur.com/4R0wOJC.jpg)  |![](https://i.imgur.com/iE3E1k3.jpg)   |![](https://i.imgur.com/k169yHN.png)    |

#### Testing


| Input |Output |
| -------- | -------- |
| ![](https://i.imgur.com/OMaPqFQ.jpg)   | ![](https://i.imgur.com/Tr5tXcb.png)    |
| ![](https://i.imgur.com/kQcqQhd.jpg) | ![](https://i.imgur.com/MrfV3FR.png)  |
| ![](https://i.imgur.com/Pk2KtB5.jpg) | ![](https://i.imgur.com/r1hcN9m.png) |
## Extension work
#### Sketch Coloring
Reverse version of previous model which converts grayscale sketch to RGB picture. The model was trained by 5000 sketch/pictures with 100 epoches.

| Input | Output |
| -------- | -------- | 
| ![](https://i.imgur.com/ww36cdS.png)    | ![](https://i.imgur.com/raHucjo.png)     | 



#### Style Changing
Combine both Anime Sketching and Sketch Coloring model, we can perform a style change on given photo.


| Input | Output |
| -------- | -------- | 
| ![](https://i.imgur.com/3OtDTEi.jpg)   |![](https://i.imgur.com/W0fbzdo.png)    | 


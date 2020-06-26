# Multi-Source Domain Adaptation

# Usage:
First, we need to  get the dataset first.

### Dataset
Use the following command we build for downloading dataset.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store it as name `data`.

### Using Pretrained Models


### Train the Model
Simply type the following command.

    python train_improved_DANN.py

There are four different case for training, if you want to switch other training case, you can alter the following lines in the code:
```python
source1 = 'quickdraw'
source2 = 'infograph'
source3 = 'real'
target = 'sketch'
```

### Prediction 
To predict the target image, type the following command that will produce `pred.csv` file

    python test.py
 
### Evaluation

To evaluate the accuracy, type the following command thae will check the accuracy based on the previous output `pred.csv`

    python check_accuracy.py

    
## Packages
Below is a list of packages we used to implement this project:

> [`CUDA`](https://developer.nvidia.com/cuda-10.1-download-archive-base): 10.1  
> [`python`](https://www.python.org/): 3.6.9 
> [`torch`](https://pytorch.org/): 1.4.0  
> [`numpy`](http://www.numpy.org/): 1.18.2  
> [`pandas`](https://pandas.pydata.org/): 0.25.1
> [`PIL`](https://pypi.org/project/Pillow/): 6.1.0
> [`torchvision`](https://pypi.org/project/torchvision/): 0.4.0
> [`cv2`](https://pypi.org/project/opencv-python/), [`matplotlib`](https://matplotlib.org/)   
> [The Python Standard Library](https://docs.python.org/3/library/)
> [`tqdm`](https://tqdm.github.io/)


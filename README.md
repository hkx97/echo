# echo
## A pipeline for automatically 2D echocardiography interpretion  
### 1.clone the code and then run run.py  
### 2.input the whole filepath. Such as D:\xx\xx\xx\filename.dicom
### 3.you can use the example.dcm for a start. Just input "./example.dcm"  
![image](https://github.com/hkx97/echo/blob/main/assessment.png)

# Performing on your own datasets  
## Different cardiac views like a2c a4c plax psax and so on are accepted.
### 
## envs:  
### 1.install anaconda python=3.7 from https://www.anaconda.com/  
### 2.conda create --name yourenvs python=3.7  ==> create a envs  
### 3.conda activate yourenvs then conda install some package as follows:  
  #### 3.1 conda install tensorflow-gpu==2.0.0 python=3.7  
  #### 3.2 pip install sklearn Augmentor pydicom imgaug pandas matplotlib h5py==2.10 opencv-python

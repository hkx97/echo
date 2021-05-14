# 1.A pipeline for automatically 2D echocardiography interpretion  
### 1.1.clone the code and then run run.py  
### 1.2.input the whole filepath. Such as D:\xx\xx\xx\filename.dicom
### 1.3.you can use the example.dcm for a start. Just input "./example.dcm" after run run.py. 



<img src="https://github.com/hkx97/echo/blob/main/assessment_a2c.png" width="400"/><img src="https://github.com/hkx97/echo/blob/main/assessment_a4c.png" width="400"/> 
    
    
    
# 2.Performing on your own datasets for training ED and ES identified model. 
### 2.1.Different cardiac views like a2c a4c plax psax and so on are accepted.
### 2.2.cd train dir and organize your own datasets and label as example description.
### 2.3.the directory names you create maybe not equal to the code, check and rectify them in code.
### 2.4.Pipeline for automatically identifying ED and ES:
<img src="https://github.com/hkx97/echo/blob/main/moxing%20.png" width="700"/>

# 3.envs:  this step is easy ~
### 3.1.install anaconda python=3.7 from https://www.anaconda.com/  
### 3.2.conda create --name yourenvs python=3.7  ==> create a envs  
### 3.3.conda activate yourenvs then conda install some package as follows:  
  #### 3.3.1 conda install tensorflow-gpu==2.0.0 python=3.7  
  #### 3.3.2 pip install sklearn Augmentor pydicom imgaug pandas matplotlib h5py==2.10 opencv-python

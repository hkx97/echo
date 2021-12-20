# 1. A webserver for automatically 2D echocardiography interpretion. http://echoo.natapp1.cc
## function include:
### 1.1 cardiac views identify
### 1.2 cardiac parameter compute
### 1.3 LVEF measurement
### 1.4 cardiac cycle identify (ED、 ES)
## For an example:
### input "./example.dcm" after exec run.py. 

<img src="https://github.com/hkx97/echo/blob/main/assessment_a2c.png" width="400"/><img src="https://github.com/hkx97/echo/blob/main/assessment_a4c.png" width="400"/> 

# 2. Train the models on your own datasets. 
### 2.1.Different cardiac views like a2c a4c plax psax and so on are accepted for training.
### 2.2.cd train dir and normalize your datasets as example description.
### 2.3.the directory names you create maybe not equal to the code, check and rectify them in code.
### 2.4.Pipeline for automatically identifying ED and ES:
<img src="https://github.com/hkx97/echo/blob/main/moxing%20.png" width="700"/>

# 3. envs and requirements:  
### 3.1.install anaconda python=3.7 from https://www.anaconda.com/  
### 3.2.conda create --name yourenvs python=3.7  ==> create a envs  
### 3.3.conda activate yourenvs then conda install some package as follows:  
  #### 3.3.1 conda install tensorflow-gpu==2.0.0 python=3.7  (cpu is also available)
  #### 3.3.2 pip install sklearn Augmentor pydicom imgaug pandas matplotlib h5py==2.10 opencv-python

# 4. deployment
### we use python flask web server for deployment.
### local test by exec main.py, then the server will run on 127.0.0.1:5000.
### support deployment on cpu.  


# 5. future plan:
### support more cardiac views cls.
### deployment by openvino.

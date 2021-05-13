import load_Comparison_model
import data_split
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow import keras
from evaluate import *
from sklearn.metrics import roc_curve, auc
from load_data import *
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def init_dir():
    path = "../"
    import shutil
    for top in ("trainp1", "trainp2", "valp1", "valp2", "testp1", "testp2"):
        if os.path.exists(path+top):
            shutil.rmtree(path=path+top)


def init_config():
    print("please input cardiac view (2 or 4):")
    qiemian = int(input())
    print("please input image_size:")
    m = int(input())
    print("please input train_epochs:")
    epochs = int(input())
    print("please input batch_size:")
    batch_size = int(input())
    return qiemian, m, epochs, batch_size


def init_dataset(filepath, labelpath, train_sample, test_sample, val_sample, qiemian, m):

    ground_truth = get_label(labelpath)
    trainp1, trainp2, trainlabel = load_datasets(filepath, train_sample, ground_truth, mark=1, a4c_or_a2c=qiemian, m=m)
    testp1, testp2, testlabel = load_datasets(filepath, test_sample, ground_truth, mark=0, a4c_or_a2c=qiemian, m=m)
    valp1, valp2, vallabel = load_datasets(filepath, val_sample, ground_truth, mark=2, a4c_or_a2c=qiemian, m=m)
    augment()
    trainp1_aug, trainp2_aug = load_aug_data(path='../trainp1/', m=m)
    trainp1 = np.concatenate((trainp1, trainp1_aug), axis=0)
    trainp2 = np.concatenate((trainp2, trainp2_aug), axis=0)
    trainlabel = np.concatenate((trainlabel, trainlabel), axis=0)
    trainlabel = tf.one_hot(trainlabel, depth=2)
    testlabel = tf.one_hot(testlabel, depth=2)
    vallabel = tf.one_hot(vallabel, depth=2)
    print("datasets is ready! shape is :")
    print(trainp1.shape, trainp2.shape, trainlabel.shape, testp1.shape, testp2.shape, testlabel.shape, valp1.shape,
          valp2.shape, vallabel.shape)
    return trainp1, trainp2, trainlabel, valp1, valp2, vallabel, testp1, testp2, testlabel


def train():
    # init_dir
    init_dir()
    # initial the record list
    TPR_list = []
    FPR_list = []
    auc_list = []
    acc_list = []
        
    edaFDlist ,esaFDlist, edplist, esplist, edrlist, esrlist, eddistribution, esdistribution, sampleError= [],[],[],[],[],[],[],[],[]
    # initial the config
    train_config = init_config()
    qiemian, m, epochs, batch_size = train_config
    # initial the dataset
    print("Enter T to use default data split, else F")
    key = True if input() == "T" else False
    train_sample, test_sample, val_sample = data_split.data_split(fixing_split=key)
    print("loading datasets......")

    """    
    1.
    if use your own datasets for training, you should rectify the labelpath and filepath
    in our experiments, labels are stored in xx.csv file and existed in root dir
    such as: ED  ES  ED  ES   mark (indicate the first frame is ED or ES)
    No.1      5  25  50  75    0
    No.2         10  30  50    1
    it is illustrated on a example.csv file
    2.
    The dataset is stored as follows:
    ../Frames/Patientxxxx/axc/x.png
    such as ../Frames/Patient0001/a2c/1.png
    it is illustrated on a demoDataset
    """

    labelpath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "\\a" + str(qiemian) + "c-copy.csv"
    filepath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "\\Frames\\"
    datasets = init_dataset(filepath, labelpath,train_sample,test_sample,val_sample,qiemian,m)
    trainp1, trainp2, trainlabel, valp1, valp2, vallabel, testp1, testp2, testlabel = datasets
    print("verify cardiac view----downscaling----epochs----batch_size press enter to continue else n to quit:\n",
          train_config)
    if input() == "n":
        exit()
    print("training......")

    for i in range(5):
        model = load_Comparison_model.load_model(input_size=m)
        lr_decay = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, verbose=1,
                                                     min_lr=1e-5)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)
        save_path = '../test.hdf5'
        check_points = keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True,
                                                       save_weights_only=True, mode='min')
        model.fit([trainp1, trainp2], trainlabel,
                  validation_data=[[valp1, valp2], vallabel],
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle=True,
                  callbacks=[early_stopping, check_points, lr_decay]
                  )
        model.load_weights(save_path)
        acc = model.evaluate([testp1, testp2], testlabel, verbose=0)[1]
        acc_list.append(acc)

        result = Truth_label(model, filepath, test_sample, labelpath, qiemian, m=m)
        eval_result = pingjiazhibiao(result)
        edaFDlist.append(eval_result[0][0])
        esaFDlist.append(eval_result[2][0])
        edplist.append(eval_result[0][1])
        esplist.append(eval_result[2][1])
        edrlist.append(eval_result[0][2])
        esrlist.append(eval_result[2][2]) 
        eddistribution.append(eval_result[1])
        esdistribution.append(eval_result[3])
        sampleError.append(eval_result[4])


                         
         
        FPR, TPR, threshold = roc_curve(testlabel[:, 0], model.predict([testp1, testp2])[:, 0], pos_label=1)

        auc_list.append(auc(FPR, TPR))
        FPR_list.append(FPR)
        TPR_list.append(TPR)

    """
    compute the mean error distribution of 5 experiments
    error distributes from -5 to 5
    if >5 or <-5 ,it will be counted into 6 
    """
    total1 = 0
    total2 = 0
    edres,esres = {},{}
    for key in range(-5,7):
        for m in range(5):
            total1+=eddistribution[m][key]
            total2+=esdistribution[m][key]
        edres[key] = total1/5
        esres[key] = total2/5
        total1 = 0
        total2 = 0

    resultDict = {}
    nameList = ["edaFDlist" ,"esaFDlist", "edplist", "esplist", "edrlist", "esrlist","acc_list","auc_list","sampleError"]
    for number,names in  enumerate ([edaFDlist ,esaFDlist, edplist, esplist, edrlist, esrlist,acc_list,auc_list,sampleError]):
        resultDict[nameList[number][:-4]] = meanAndSd(names)

    """
    At last , the five experiments are completed. mean±sd will be considered.
    The accuracy and auc for binary-cls model will be computed.
    The metrics for ED and ES include precision, recall, aFD(equal to mae) will be computed respectively.
    The sampleError will be computed if model has not worked on a certain one.  
    """

    f = open("./result/"+str(train_config)+str(len(train_sample))+"eval_result.txt", "w", encoding="utf-8")
    f.write(str(resultDict))
    f.write("===================================")
    f.write(str(edres))
    f.write("===================================")
    f.write(str(esres))  
    f.write("===================================")
    f.write(str(FPR_list))
    f.write("===================================")
    f.write(str(TPR_list))                     
    f.close()
    print("Training process is done!")


if __name__ == "__main__":
    train()



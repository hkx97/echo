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

# clean the cache dir
def init_dir():
    path = "../"
    import shutil
    for top in ("trainp1", "trainp2", "valp1", "valp2", "testp1", "testp2"):
        if os.path.exists(path+top):
            shutil.rmtree(path=path+top)

# init train config
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

# init the dataset
def init_dataset(filepath, labelpath, train_sample, test_sample, val_sample, qiemian, m):
    """
    1.get the ground_truth
    2.then generate the datasets
    3.augment and store aug_data before training
    4.read the aug_data then concat it with the existing datasets
    5.prepare the label and encode it using one-hot
    """
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
    # we have many index for ecaluating: aFD precision recall error distribution sampleError
    edaFDlist ,esaFDlist, edplist, esplist, edrlist, esrlist, eddistribution, esdistribution, sampleError= [],[],[],[],[],[],[],[],[]
    # initial the config
    train_config = init_config()
    # some hypeparameters:
    # qiemian(2 or 4) is equal to cardiac view. such as a2c or a4c.
    # m is downsampling size. we set it 128 in our experiments.
    # epochs: we set it 100 and it's enough for early stopping work.
    #batch_size = 32 here
    qiemian, m, epochs, batch_size = train_config
    # initial the dataset
    # we split the dataset as: train(60) : val(15) : test(25)
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
    # labelpath and filepath
    labelpath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "\\a" + str(qiemian) + "c-copy.csv"
    filepath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "\\Frames\\"
    datasets = init_dataset(filepath, labelpath,train_sample,test_sample,val_sample,qiemian,m)
    trainp1, trainp2, trainlabel, valp1, valp2, vallabel, testp1, testp2, testlabel = datasets
    # verify some hypeparameters
    print("verify cardiac view----downscaling----epochs----batch_size press enter to continue else n to quit:\n",
          train_config)
    if input() == "n":
        exit()
    print("training......")
    # five individual experiments are performed. model will be initialized by the same way each time.
    for i in range(5):
        # init the model
        model = load_Comparison_model.load_model(input_size=m)
        # learning rate decay strategy
        lr_decay = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, verbose=1,
                                                     min_lr=1e-5)
        # earlystopping strategy
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)
        # model weights' save_path
        save_path = '../test.hdf5'
        # save the best weights of lowest val_loss
        check_points = keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True,
                                                       save_weights_only=True, mode='min')
        # training...
        model.fit([trainp1, trainp2], trainlabel,
                  validation_data=[[valp1, valp2], vallabel],
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle=True,
                  callbacks=[early_stopping, check_points, lr_decay]
                  )
        # training is completed. then load the saved best weights for evaluating.
        model.load_weights(save_path)
        # evaluating the accuracy on testset
        acc = model.evaluate([testp1, testp2], testlabel, verbose=0)[1]
        acc_list.append(acc)
        # computing the aFD presicion recall and so on. result are restored in eval_result.
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


                         
        # FPR TPR for ROC. auc:area under the curve
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
    # using a dict for ultimate result
    resultDict = {}
    nameList = ["edaFDlist" ,"esaFDlist", "edplist", "esplist", "edrlist", "esrlist","acc_list","auc_list","sampleError"]
    # meanAndSd for averaging of 5 experiment.
    for number,names in  enumerate ([edaFDlist ,esaFDlist, edplist, esplist, edrlist, esrlist,acc_list,auc_list,sampleError]):
        resultDict[nameList[number][:-4]] = meanAndSd(names)

    """
    At last , the five experiments are completed. mean±sd will be considered.
    The accuracy and auc for binary-cls model will be computed.
    The metrics for ED and ES include precision, recall, aFD(equal to mae) will be computed respectively.
    The sampleError will be computed if model has not worked on some certain cases.  
    """
    # all result will be writed into a .txt file
    f = open("./result/"+str(train_config)+str(len(train_sample))+"eval_result.txt", "w", encoding="utf-8")
    f.write(str(resultDict)+"\n")
    f.write("==================================="+"\n")
    f.write(str(edres)+"\n")
    f.write("==================================="+"\n")
    f.write(str(esres)+"\n")  
#     f.write("==================================="+"\n")
#     f.write(str(FPR_list)+"\n")
#     f.write("==================================="+"\n")
#     f.write(str(TPR_list))                     
    f.close()
    print("Training process is done!")


if __name__ == "__main__":
    train()



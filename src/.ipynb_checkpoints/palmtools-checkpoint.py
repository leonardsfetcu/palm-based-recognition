import matplotlib.pyplot as plt
import os
import os.path
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import const
import random
import shutil


def FYODB(i):
    switcher={
        1: "ROI",
        2: "ORIGINAL",
        3: "GENERATED",
        4: "PALM",
        5: "DORSAL",
        6: "WRIST"
    }
    return switcher.get(i,"Invalid selection!")

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def loadHistory(history_path):
    return pickle.load(open(history_path), "rb")

def saveHistory(history_path, history):
    with open(history_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

def getNoSamples(category,biometric):
    count = 0
    s1_path = ''
    s2_path = ''
############################################################################################
    if category == const.ROI:
        if biometric == const.PALM:
            palm_roi_s1_dir = const.ROI_DIR + "Session1/Palm/"
            palm_roi_s2_dir = const.ROI_DIR + "Session2/Palm/"
            count =  len([1 for x in list(os.scandir(palm_roi_s1_dir)) if x.is_file()])
            count += len([1 for x in list(os.scandir(palm_roi_s2_dir)) if x.is_file()])
            s1_path = palm_roi_s1_dir
            s2_path = palm_roi_s2_dir
        if biometric == const.WRIST:
            wrist_roi_s1_dir = const.ROI_DIR + "Session1/Wrist/"
            wrist_roi_s2_dir = const.ROI_DIR + "Session2/Wrist/"
            count =  len([1 for x in list(os.scandir(wrist_roi_s1_dir)) if x.is_file()])
            count += len([1 for x in list(os.scandir(wrist_roi_s2_dir)) if x.is_file()])
            s1_path = wrist_roi_s1_dir
            s2_path = wrist_roi_s2_dir
        if biometric == const.DORSAL:
            dorsal_roi_s1_dir = const.ROI_DIR + "Session1/Dorsal/"
            dorsal_roi_s2_dir = const.ROI_DIR + "Session2/Dorsal/"
            count =  len([1 for x in list(os.scandir(dorsal_roi_s1_dir)) if x.is_file()])
            count += len([1 for x in list(os.scandir(dorsal_roi_s2_dir)) if x.is_file()])
            s1_path = dorsal_roi_s1_dir
            s2_path = dorsal_roi_s2_dir
##############################################################################################            
    if category == const.ORIGINAL:
        if biometric == const.PALM:
            palm_original_s1_dir = const.ORIGINAL_DIR + "Session1/Palm/"
            palm_original_s2_dir = const.ORIGINAL_DIR + "Session2/Palm/"
            count =  len([1 for x in list(os.scandir(palm_original_s1_dir)) if x.is_file()])
            count += len([1 for x in list(os.scandir(palm_original_s2_dir)) if x.is_file()])
            s1_path = palm_original_s1_dir
            s2_path = palm_original_s2_dir
        if biometric == const.WRIST:
            wrist_original_s1_dir = const.ORIGINAL_DIR + "Session1/Wrist/"
            wrist_original_s2_dir = const.ORIGINAL_DIR + "Session2/Wrist/"
            count =  len([1 for x in list(os.scandir(wrist_original_s1_dir)) if x.is_file()])
            count += len([1 for x in list(os.scandir(wrist_original_s2_dir)) if x.is_file()])
            s1_path = wrist_original_s1_dir
            s2_path = wrist_original_s2_dir
        if biometric == const.DORSAL:
            dorsal_original_s1_dir = const.ORIGINAL_DIR + "Session1/Dorsal/"
            dorsal_original_s2_dir = const.ORIGINAL_DIR + "Session2/Dorsal/"
            count =  len([1 for x in list(os.scandir(dorsal_original_s1_dir)) if x.is_file()])
            count += len([1 for x in list(os.scandir(dorsal_original_s2_dir)) if x.is_file()])      
            s1_path = dorsal_original_s1_dir
            s2_path = dorsal_original_s2_dir
###############################################################################################            
    if category == const.GENERATED:
        if biometric == const.PALM:
            palm_generated_s1_dir = const.GENERATED_DIR + "Session1/Palm/"
            palm_generated_s2_dir = const.GENERATED_DIR + "Session2/Palm/"
            count =  len([1 for x in list(os.scandir(palm_generated_s1_dir)) if x.is_file()])
            count += len([1 for x in list(os.scandir(palm_generated_s2_dir)) if x.is_file()])            
            s1_path = palm_generated_s1_dir
            s2_path = palm_generated_s2_dir
        if biometric == const.WRIST:
            wrist_generated_s1_dir = const.GENERATED_DIR + "Session1/Wrist/"
            wrist_generated_s2_dir = const.GENERATED_DIR + "Session2/Wrist/"
            count =  len([1 for x in list(os.scandir(wrist_generated_s1_dir)) if x.is_file()])
            count += len([1 for x in list(os.scandir(wrist_generated_s2_dir)) if x.is_file()])            
            s1_path = wrist_generated_s1_dir
            s2_path = wrist_generated_s2_dir
        if biometric == const.DORSAL:
            dorsal_generated_s1_dir = const.GENERATED_DIR + "Session1/Dorsal/"
            dorsal_generated_s2_dir = const.GENERATED_DIR + "Session2/Dorsal/"
            count =  len([1 for x in list(os.scandir(dorsal_generated_s1_dir)) if x.is_file()])
            count += len([1 for x in list(os.scandir(dorsal_generated_s2_dir)) if x.is_file()])     
            s1_path = dorsal_generated_s1_dir
            s2_path = dorsal_generated_s2_dir
###############################################################################################            
    return count, s1_path, s2_path
    
def getFilesArray(category, biometric):
    files = []
    dataset_dir = ''
    s1_path = ''
    s2_path = ''
    count = 0

        
    count, s1_path, s2_path = getNoSamples(category,biometric)
    for file in os.listdir(s1_path):
        if not "ipynb" in file:
            if ".png" or ".jpg" in file:
                
                splitted = file.split("_")
                if category == const.GENERATED:
                    personId = splitted[0].split("s")[1]
                    hand = splitted[2]
                    
                if category == const.ROI:
                    personId = splitted[0]
                    hand = splitted[1].split(".")[0]
        
                _class = personId + '_' + hand
                files.append([s1_path, file, "s1.",_class,category,biometric,const.S1])
                       
    for file in os.listdir(s2_path):
        if not "ipynb" in file:
            if ".png" or ".jpg" in file:
                splitted = file.split("_")
                if category == const.GENERATED:
                    personId = splitted[0].split("s")[1]
                    hand = splitted[2]
                    
                if category == const.ROI:
                    personId = splitted[0]
                    hand = splitted[1].split(".")[0]
        
                _class = personId + "_" + hand
                files.append([s2_path, file, "s2.",_class,category,biometric,const.S2])
    
    return files
    
def getDirArchitecture(biometric):
    
    dataset_dir = ""
    
    if biometric == const.PALM:
        dataset_dir = const.PALM_DS_DIR
       
    if biometric == const.WRIST:
        dataset_dir = const.WRIST_DS_DIR
     
    if biometric == const.DORSAL:
        dataset_dir = const.DORSAL_DS_DIR
    
    return dataset_dir

def createDirectoryArch(biometric):
    
    # Obtinere cale catre directorul cu samples
    dataset_dir = getDirArchitecture(biometric)
    
    # Concatenare cale catre subdirectoarele directorului cu samples
    train_dir = dataset_dir + const.TRAIN_DIR
    valid_dir = dataset_dir + const.VALID_DIR
    test_dir = dataset_dir + const.TEST_DIR
    all_dir = dataset_dir + const.ALL_DIR
        
    # Verificare si creare director radacina pt setul de date
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    # Verificare si creare directoare de TRAIN, VALIDATE, TEST si ALL
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(valid_dir):
        os.mkdir(valid_dir)
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)
    if not os.path.isdir(all_dir):
        os.mkdir(all_dir)

    # Verificare si creare directoare pentru fiecare clasa de imagini
    for _class in range(1,161):

        train_class_L = train_dir + str(_class) + "_L/"
        train_class_R = train_dir + str(_class) + "_R/"
        valid_class_L = valid_dir + str(_class) + "_L/"
        valid_class_R = valid_dir + str(_class) + "_R/"
        test_class_L = test_dir + str(_class) + "_L/"
        test_class_R = test_dir + str(_class) + "_R/"
        all_class_L = all_dir + str(_class) + "_L/"
        all_class_R = all_dir + str(_class) + "_R/"

        if not os.path.isdir(train_class_L):
            os.mkdir(train_class_L)
        if not os.path.isdir(train_class_R):
            os.mkdir(train_class_R)
            
        if not os.path.isdir(valid_class_L):
            os.mkdir(valid_class_L)
        if not os.path.isdir(valid_class_R):
            os.mkdir(valid_class_R)
            
        if not os.path.isdir(test_class_L):
            os.mkdir(test_class_L)
        if not os.path.isdir(test_class_R):
            os.mkdir(test_class_R)
            
        if not os.path.isdir(all_class_L):
            os.mkdir(all_class_L)
        if not os.path.isdir(all_class_R):
            os.mkdir(all_class_R)
            
    
    
def createDataset(biometric, files, train_ratio=0, valid_ratio=0, test_ratio=0):
    
    
    dataset_dir = getDirArchitecture(biometric)
    
    if os.path.isdir(dataset_dir + const.ALL_DIR):
        shutil.rmtree(dataset_dir + const.ALL_DIR)
    
    createDirectoryArch(biometric)

#['/home/leonard/Desktop/PyProjects/palm-cnn/FYODB/Generated_Images/Session1/Palm/',
# 's63_9_L_S1.jpg',
# 's1.',
# '63_L',
# 33,
# 44,
# 101]
 
    # Copierea sample-urilor din baza de date in setul de date complet (All)
    for file in files:
        file_path = file[0]
        file_name = file[1]
        file_session = file[2].split('.')[0]
        _class = file[3]
        src_path = file_path + file_name
        dst_path = dataset_dir + const.ALL_DIR + _class + "/" + file_session + "_" + file_name
        if os.path.isfile(src_path) and (".jpg" in file_name or ".png" in file_name):
            shutil.copy(src_path, dst_path)
    
    
    # Copiere sample-uri din setul de date complet (All) in directoarele
    # aferente seturilor de date de antrenare, validare si testare conform proportiilor
    for _class in range(1,161):
        clear_dir_files = []
        src_file_dir_L = dataset_dir + const.ALL_DIR + str(_class) + "_L/"
        src_file_dir_R = dataset_dir + const.ALL_DIR + str(_class) + "_R/"
        
        src_file_path_L = [src_file_dir_L + file for file in os.listdir(src_file_dir_L) if (not "ipynb" in file) and (".png" or ".jpg" in file)]
        src_file_path_R = [src_file_dir_R + file for file in os.listdir(src_file_dir_R) if (not "ipynb" in file) and (".png" or ".jpg" in file)]
        
        random.shuffle(src_file_path_L)
        random.shuffle(src_file_path_R)
    
        ### Training samples ###       
        for file in src_file_path_L[:int(train_ratio * len(src_file_path_L))]:
            if os.path.isfile(file) and (".jpg" in file or ".png" in file):
                shutil.copy(file, dataset_dir + const.TRAIN_DIR + str(_class) + "_L/")
                
        for file in src_file_path_R[:int(train_ratio * len(src_file_path_R))]:
            if os.path.isfile(file) and (".jpg" in file or ".png" in file):
                shutil.copy(file, dataset_dir + const.TRAIN_DIR + str(_class) + "_R/")

        ### Validation samples ###       
        for file in src_file_path_L[int(train_ratio * len(src_file_path_L)):int((train_ratio + valid_ratio) * len(src_file_path_L))]:
            if os.path.isfile(file) and (".jpg" in file or ".png" in file):
                shutil.copy(file, dataset_dir + const.VALID_DIR + str(_class) + "_L/")
       
        for file in src_file_path_R[int(train_ratio * len(src_file_path_R)):int((train_ratio + valid_ratio) * len(src_file_path_R))]:
            if os.path.isfile(file) and (".jpg" in file or ".png" in file):
                shutil.copy(file, dataset_dir + const.VALID_DIR + str(_class) + "_R/")

        ### Testing samples ###     
        for file in src_file_path_L[int((train_ratio + valid_ratio) * len(src_file_path_L)):int((train_ratio + valid_ratio + test_ratio) * len(src_file_path_L))]:
            if os.path.isfile(file) and (".jpg" in file or ".png" in file):
                shutil.copy(file, dataset_dir + const.TEST_DIR + str(_class) + "_L/")
        
        for file in src_file_path_R[int((train_ratio + valid_ratio) * len(src_file_path_R)):int((train_ratio + valid_ratio + test_ratio) * len(src_file_path_R))]:
            if os.path.isfile(file) and (".jpg" in file or ".png" in file):
                shutil.copy(file, dataset_dir + const.TEST_DIR + str(_class) + "_R/")                
        


def plotModelStats(modelHistory):
    acc = modelHistory.history['accuracy']
    val_acc = modelHistory.history['val_accuracy']
    loss = modelHistory.history['loss']
    val_loss = modelHistory.history['val_loss']
    epochs = range(1,len(acc)+1)
    
    plt.plot(epochs,acc,'bo',label='Training acc')
    plt.plot(epochs,val_acc,'b',label='Validation acc')
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()

    plt.plot(epochs,loss,'bo',label='Training loss')
    plt.plot(epochs,val_loss,'b',label='Validation loss')
    plt.title("Training and validation loss")
    plt.legend()

    plt.show()

def datasetCounter(trainDir,validDir,testDir):
    count_train = 0
    count_test = 0
    count_valid = 0
    for subject in range(1,161):
        count_train += len(os.listdir(trainDir+"s"+str(subject)))
        count_test += len(os.listdir(testDir+"s"+str(subject)))
        count_valid += len(os.listdir(validDir+"s"+str(subject)))
    print("Train dateset: ", count_train)
    print("Test dateset: ", count_test)
    print("Validation dateset: ", count_valid)

    
def extractFeatures(model, generator, out_shape, batch_size):
#    extractFeatures(conv_base_resnet,
#                   palm_train_generator,
#                   (6400*train_ratio,) + resnet_out_shape,
#                   batch_size
#                  )

    
    features = np.zeros(shape=out_shape)
    labels = np.zeros(shape=(out_shape[0],160))
    

    i = 0
    for inputs_batch,labels_batch in generator:
        
        features_batch = model.predict(inputs_batch)
        
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i* batch_size : (i + 1) * batch_size] = labels_batch
        
        i += 1
        if i * batch_size >= out_shape[0]:
            break
            
    return features, labels

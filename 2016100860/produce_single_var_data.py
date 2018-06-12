
# coding: utf-8

# In[17]:



# coding: utf-8
# encoding=utf8  

# In[6]:

#dc = 0 表示按用户划分训练集测试集 dc=1表示按时间划分训练集测试集
#scaler = 1 表示标准化 =2表示归一化 scaler = 0 表示不做预处理
def produce_single_var_data(ntime, TJ_dict, dc=0, scaler="mm", trainf =  "train_3_bph_180403.txt", testf = "test_3_bph_180403.txt", y = "bph", homo=False):
    import json
    import numpy as np
    from datetime import datetime
    import pickle
   
    TJ_ntime_dict = {}
    for key, value in TJ_dict.items():
        if(TJ_dict[key]['TJ_items']["times"] >= ntime):
            TJ_ntime_dict[key] = value



    from keras.preprocessing.sequence import pad_sequences
    ## 先把训练集测试集分出来，再进行padding，这里的y_train只有最后一个, y_test也只有最后一个
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    if(dc==0):
        with open(trainf, "rb") as fp:   # Unpickling
            train = pickle.load(fp)
        with open(testf, "rb") as fp:   # Unpickling
            test = pickle.load(fp)
        
        train = [int(x) for x in train]
        test = [int(x) for x in test]
        
        data_train_ntime_var = []
        data_test_ntime_var = []
        train_context = []
        test_context = []
        
        for key in train:
            key=str(key)
            times=TJ_ntime_dict[key]['TJ_items']['times']
            for i in range(times):
                if(homo):
                    if(i>ntime-1):
                        continue
                TJ_ntime_dict[key]['TJ_items']['items'][i]["年龄"] = int(TJ_ntime_dict[key]['TJ_items']['items'][i]["年龄"])
                if("体检时间" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["体检时间"]
                if("身份证" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["身份证"]
                if("用户ID" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["用户ID"]
                if("工作单位名称" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["工作单位名称"]
                if("手机号码" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["手机号码"]
                if("年龄" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["年龄"]
                if("用户姓名" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["用户姓名"]
            if(homo):
                data_train_ntime_var.append([list(x.values()) for x in TJ_ntime_dict[key]['TJ_items']['items'][0:ntime]])
            else:
                data_train_ntime_var.append([list(x.values()) for x in TJ_ntime_dict[key]['TJ_items']['items']])
            train_context.append([TJ_ntime_dict[key]["性别"],TJ_ntime_dict[key]["省份_北京"],TJ_ntime_dict[key]["省份_海南"], TJ_ntime_dict[key]["年龄"]])
        
        for key in test:
            key=str(key)
            times=TJ_ntime_dict[key]['TJ_items']['times']
            for i in range(times):
                if(homo):
                    if(i>ntime-1):
                        continue
                TJ_ntime_dict[key]['TJ_items']['items'][i]["年龄"] = int(TJ_ntime_dict[key]['TJ_items']['items'][i]["年龄"])
                if("体检时间" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["体检时间"]
                if("身份证" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["身份证"]
                if("用户ID" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["用户ID"]
                if("工作单位名称" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["工作单位名称"]
                if("手机号码" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["手机号码"]
                if("年龄" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["年龄"]
                if("用户姓名" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["用户姓名"]
            if(homo):
                data_test_ntime_var.append([list(x.values()) for x in TJ_ntime_dict[key]['TJ_items']['items'][0:ntime]])
            else:
                data_test_ntime_var.append([list(x.values()) for x in TJ_ntime_dict[key]['TJ_items']['items']])
            #print(TJ_ntime_dict[key]['TJ_items']['items'][0].keys())
            test_context.append([TJ_ntime_dict[key]["性别"],TJ_ntime_dict[key]["省份_北京"],TJ_ntime_dict[key]["省份_海南"],TJ_ntime_dict[key]["年龄"]])

        for user in data_train_ntime_var:
            #user.reverse()
            x_train.append(user[0:-1])
            if(y=="bph"):
                y_train.append(user[-1][0])
            elif(y=="bpl"):
                y_train.append(user[-1][2])
                
        for user in data_test_ntime_var:
            #user.reverse()
            x_test.append(user[0:-1])
            if(y=="bph"):
                y_test.append(user[-1][0])
            elif(y=="bpl"):
                y_test.append(user[-1][2])
        
    elif(dc==1):
        data_ntime_var = []
        data_context = []
        for key in TJ_ntime_dict:
            times=TJ_ntime_dict[key]['TJ_items']['times']
            for i in range(times):
                if(homo):
                    if(i>ntime-1):
                        continue
                TJ_ntime_dict[key]['TJ_items']['items'][i]["年龄"] = int(TJ_ntime_dict[key]['TJ_items']['items'][i]["年龄"])
                if("体检时间" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["体检时间"]
                if("身份证" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["身份证"]
                if("用户ID" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["用户ID"]
                if("工作单位名称" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["工作单位名称"]
                if("手机号码" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["手机号码"]
                if("年龄" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["年龄"]
                if("用户姓名" in TJ_ntime_dict[key]['TJ_items']['items'][i]):
                    del TJ_ntime_dict[key]['TJ_items']['items'][i]["用户姓名"]
            if(homo):
                data_ntime_var.append([list(x.values()) for x in TJ_ntime_dict[key]['TJ_items']['items'][0:ntime]])
            else:
                data_test_ntime_var.append([list(x.values()) for x in TJ_ntime_dict[key]['TJ_items']['items']])
            data_context.append([TJ_ntime_dict[key]["性别"],TJ_ntime_dict[key]["省份_北京"],TJ_ntime_dict[key]["省份_海南"],TJ_ntime_dict[key]["年龄"]])
        
        for user in data_ntime_var:
            #user.reverse()
            x_train.append(user[0:-2])
            #y_train.append([tj[1] for tj in user[1:-1]])
            if(y == "bph"):
                y_train.append(user[-2][0])
            elif(y == "bpl"):
                y_train.append(user[-2][2])
            x_test.append(user[0:-1])
            #y_test.append([tj[1] for tj in user[1:]])
            if(y == "bph"):
                y_test.append(user[-1][0])
            elif(y == "bpl"):
                y_test.append(user[-1][2])
        
        train_context = data_context
        test_context = data_context
    
    #y_train = np.array([float(x) for x in y_train])
    #y_test = np.array([float(x) for x in y_test])
    # standardize
    if scaler == "std":
        from sklearn.preprocessing import StandardScaler
        x_train, x_test = rnn_standardize(x_train, x_test)
        scaler = StandardScaler()
    elif scaler == "mm":
        from sklearn.preprocessing import MinMaxScaler
        x_train, x_test = rnn_maxminto01(x_train, x_test)
        scaler = MinMaxScaler()
    train_context=np.array(train_context, dtype="float")
    test_context=np.array(test_context, dtype="float")
    scaler.fit(train_context[:,3].reshape(-1, 1))  # Don't cheat - fit only on training data
    train_context[:,3] = scaler.transform(train_context[:,3].reshape(1, -1))
    test_context[:,3] = scaler.transform(test_context[:,3].reshape(1, -1))  # apply same transformation to test data
    
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    
    x_train = pad_sequences(x_train, value=0.0, dtype='float32', padding='post', maxlen=10)
    y_train = y_train.reshape((y_train.shape[0], 1))
    x_test = pad_sequences(x_test, value=0.0, dtype='float32', padding='post', maxlen=10)
    y_test = y_test.reshape((y_test.shape[0], 1))

    return x_train, y_train, x_test, y_test, train_context, test_context


# In[7]:

def rnn_maxminto01(train, test):
    from sklearn.preprocessing import MinMaxScaler
    from math import sqrt
    import operator
    import numpy as np
    
    train_flatten = []
    for x in train:
        for y in x:
            train_flatten.append(y)
    
    scaler = MinMaxScaler()
    scaler = scaler.fit(train_flatten)
    
    train_s = []
    test_s = []
    
    smin = scaler.data_min_
    smax = scaler.data_max_
    
    for i in range(len(train)):
        train_s.append([])
        for j in range(len(train[i])):
            tmp = train[i][j]
            tmp = [float(i) for i in tmp]
            L1 = np.array(tmp) - smin 
            L2 = smax - smin
            train_s[i].append(list(map(operator.truediv, L1, L2)))
    
    for i in range(len(test)):
        test_s.append([])
        for j in range(len(test[i])):
            tmp = test[i][j]
            tmp = [float(i) for i in tmp]
            L1 = np.array(tmp) - smin
            L2 = smax - smin
            test_s[i].append(list(map(operator.truediv, L1, L2)))
    
    return train_s, test_s


# In[8]:

def rnn_standardize(train, test):
    from sklearn.preprocessing import StandardScaler
    from math import sqrt
    import operator
    import numpy as np
    
    train_flatten = []
    for x in train:
        for y in x:
            train_flatten.append(y)
    
    scaler = StandardScaler()
    scaler = scaler.fit(train_flatten)
    
    train_s = []
    test_s = []
    
    for i in range(len(train)):
        train_s.append([])
        for j in range(len(train[i])):
            tmp = train[i][j]
            tmp = [float(i) for i in tmp]
            
            L1 = np.array(tmp) - scaler.mean_
            L2 = [sqrt(x) for x in scaler.var_]
            train_s[i].append(list(map(operator.truediv, L1, L2)))
    
    for i in range(len(test)):
        test_s.append([])
        for j in range(len(test[i])):
            tmp = test[i][j]
            tmp = [float(i) for i in tmp]
            L1 = np.array(tmp) - scaler.mean_
            L2 = [sqrt(x) for x in scaler.var_]
            test_s[i].append(list(map(operator.truediv, L1, L2)))
    
    return train_s, test_s

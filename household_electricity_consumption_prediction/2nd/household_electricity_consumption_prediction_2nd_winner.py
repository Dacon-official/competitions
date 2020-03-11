## Dacon 11회 에너지 빅데이터 활용 데이터 사이언스 콘테스트,
## Saint (팀명),
## 2019년 10월 28일 (제출날짜)

# -*- coding: utf-8 -*-
# 작업 환경: Spyder (python 3.7)
#------------------------------------------------------------------------------
# <팀명>: Saint
# 팀원: 핥핥핥핥
# 팀원: 윤지석
# 팀원: Jun
#------------------------------------------------------------------------------
#============================<Coding 구성>=======================================
# [Local Function #1]: SAMPE calculation module
# [Local Function #2]: AR_data_set calculation module
# [Local Function #3]: AR_day_set calculation module
# [Local Function #4]: linear_prediction calculation module
# [Local Function #5]: Random forest module
# [Local Function #6]: DNN module

# [Main function]
# [Section #1]: Data loading section
# [Section #2]: Data generation for training set
# [Section #3]: Anormaly detection using AR model 
# [Section #4]: data prediction for hour profile
# [Section #5]: data prediction for day profile
# [Section #6]: data prediction for month profile
#=============================================================================

import pandas as pd             #데이터 전처리
import numpy as np              #데이터 전처리
import os
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#%% Local function <SAMPE calculation module>
#------------------------------------------------------------------------------
# [Input]
# <A> : Real data.
# <F>: Forecasting data.
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------
# [Output]
# smape result.
#------------------------------------------------------------------------------ 

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


#%% Local function <AR_data_set calculation module>
# 시간 데이터 예측을 위한 데이터 셋 추출 함수.
# test.csv 파일 내에 전력 데이터를 요일 타입을 고려하여 분류함.
# 전날 데이터를 학습하여 다음날을 예측하는 방식을 사용하며, 요일 타입은 2가지로 
# 분류함. 월~금은 Workday, 토~일은 weekend    
#------------------------------------------------------------------------------
# [Input]
# <Data> : test.csv.
# <place_id>: power meter ID.
# <prev_type>: 예측 전 날의 데이터 타입(<1>:Workday <2>:Weekend)
# <Curr_type>: 예측 날의 데이터 타입(<1>:Workday <2>:Weekend)  
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------
# [Output]
# <TrainAR>: 예측 전날의 data set [day x time]
# <TestAR>: 예측 날의 data set [day x time]
#------------------------------------------------------------------------------ 
def AR_data_set(Data, place_id, prev_type, Curr_type):
    
    # Mon: 0 ~ Sun:6

    TrainAR = []; TestAR = []
    len_bad = 20  # 하루 내 NaN의 개수 기준, (24-len_bad)보다 많으면 그 날은 제거 
    Power = Data[place_id].iloc  # test.csv에서 특정 id의 전력 데이터   
    Date = Data[place_id].index  # test.csv에서 특정 id의 날짜 데이터 
    prev_aloc = [0]*24;  curr_aloc = [0]*24  # pre-allocation
    
    for ii in range(24,len(Date)):
        
        
        if (Date[ii].hour == 0) & (ii >48) & (np.sum(curr_aloc)!=24*curr_aloc[1])& (np.sum(prev_aloc)!=24*prev_aloc[1]):
            prev_idx = 0;  curr_idx = 0      # bad data idx
            
            for kk in range(0,24):
                if prev_aloc[kk]>-20:        # check the bad data.
                    prev_idx =prev_idx+1     
                else:                        # interpolate the bad data.
                    # bad data일 경우, 앞뒤로 20개의 포인트를 가져와서 
                    # interpolation 진행.
                    temp = np.zeros([1,41])
                    for qq in range(0,41):
                        temp[0,qq] = Power[(ii-24)-(24-kk)-20+qq]
                    
                    temp_temp = pd.DataFrame(data = temp)

                    temp = temp_temp.interpolate('spline',order =1)
                    temp = temp.values
                    prev_aloc[kk] = temp[0,20]
                    

            for kk in range(0,24):
                if curr_aloc[kk]>-20:       # check the bad data.
                    curr_idx =curr_idx+1
                else:
                    # bad data일 경우, 앞뒤로 20개의 포인트를 가져와서 
                    # interpolation 진행.
                    temp = np.zeros([1,41])
                    for qq in range(0,41):
                        temp[0,qq] = Power[(ii)-(24-kk)-20+qq]
                    temp_temp = pd.DataFrame(data = temp)
                    
                    temp = temp_temp.interpolate('spline',order =1)
                    temp = temp.values
                    curr_aloc[kk] = temp[0,20]

            # bad data가 특정 개수 이상이면, data set에 추가하지 않는다.  
            if (prev_idx>len_bad)&(curr_idx>len_bad):
                TrainAR.append(prev_aloc)
                TestAR.append(curr_aloc)
                        
        # 0시에 하루 데이터 초기화.                     
        if Date[ii].hour == 0:
            prev_aloc = [0]*24
            curr_aloc = [0]*24
        
        # 요일 데이터 확인.
        prev_day = Date[ii-24].weekday()
        curr_day = Date[ii].weekday()
        
        # 요일 데이터 타입 분류
        # Workday(1) = day type<5(월~금)
        # Workday(2) = day type>4(토~일)
        if ((prev_type ==1)&(prev_day<5))&((Curr_type ==2)&(curr_day>4)):           
            prev_aloc[Date[ii-24].hour] = Power[ii-24]
            curr_aloc[Date[ii].hour] = Power[ii]
        
        if ((prev_type ==1)&(prev_day<5))&((Curr_type ==1)&(curr_day<5)):           
            prev_aloc[Date[ii-24].hour] = Power[ii-24]
            curr_aloc[Date[ii].hour] = Power[ii]
            
        if ((prev_type ==2)&(prev_day>4))&((Curr_type ==2)&(curr_day>4)):           
            prev_aloc[Date[ii-24].hour] = Power[ii-24]
            curr_aloc[Date[ii].hour] = Power[ii]
            
        if ((prev_type ==2)&(prev_day>4))&((Curr_type ==1)&(curr_day<5)):           
            prev_aloc[Date[ii-24].hour] = Power[ii-24]
            curr_aloc[Date[ii].hour] = Power[ii]
                
    TrainAR = np.array(TrainAR)
    TestAR = np.array(TestAR)
    return TrainAR, TestAR


#%% Local function <AR_day_set calculation module>
# 하루 사용 데이터 예측을 위한 데이터 셋 추출 함수.
# test.csv 파일 내에 전력 데이터를 요일 타입을 고려하여 분류함.
# <요일 타입>: 월 ~ 일
# Similar day approach method만을 활용할 예정이기 때문에 최근 데이터 6주만을 
# 정리함.    
#------------------------------------------------------------------------------
# [Input]
# <Data> : test.csv.
# <place_id>: power meter ID.
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------
# [Output]
# <temp_day>: 과거 하루 전력 사용량  [week number(최근 순) x day type]
#------------------------------------------------------------------------------     
def AR_day_set(data, place_id):

    Power = data[place_id].values       #전력 데이터 
    Date = data[place_id].index         #요일 데이터 
    
    temp_day = np.zeros([6, 7])         # pre-allocation for output dataset
    mon_idx = np.zeros([1, 7])          # 몇 번째 week인지 확인하는 idx
    
    for ii in range(0,len(Power)-500):
                
        idx = len(Power) - ii -1
        
        day_idx = Date[idx].weekday()       # data의 요일정보
        time_idx = Date[idx].hour           # data의 시간정보 

        if mon_idx[0, day_idx] < 6:         # 6번째 week 이상이면 추가 X
            
            if np.isnan(Power[idx]):        # bad data restortion
                res_data = np.zeros([1, 9]) 
                # 1주전, 2주전, 3주전의 같은 요일, 시간 데이터를 저장 후 mean
                res_data[0,0] = Power[idx-24*7-1]
                res_data[0,1] = Power[idx-24*7]
                res_data[0,2] = Power[idx-24*7+1]
                
                res_data[0,3] = Power[idx-48*7-1]
                res_data[0,4] = Power[idx-48*7]
                res_data[0,5] = Power[idx-48*7+1]
                
                res_data[0,6] = Power[idx-1]
                res_data[0,7] = Power[idx-3*24*7]
                res_data[0,8] = Power[idx+1]
                # 하루 사용량 저장을 위한 시간 데이터 합
                temp_day[int(round(mon_idx[0,day_idx])), day_idx] = temp_day[int(round(mon_idx[0,day_idx])), day_idx]+ np.nanmean(res_data)
                
            else:
                # 하루 사용량 저장을 위한 시간 데이터 합
                temp_day[int(round(mon_idx[0,day_idx])), day_idx] = temp_day[int(round(mon_idx[0,day_idx])), day_idx] + Power[idx]

                
            if time_idx == 0:
                # 요일이 지나면, week확인 idx +1
                mon_idx[0,day_idx] = mon_idx[0,day_idx] + 1
                
    return temp_day




#%% Local function <linear_prediction calculation module>
# 선형 예측 방식 구현. (Autoregressive model)
# Y(예측 날) = A(coefficient)*X(예측전날)
# X-1(역행렬)*Y = A
# A를 추출하여, 예측에 활용함.     
#------------------------------------------------------------------------------
# [Input]
# <trainAR>: test.csv의 예측 전날 데이터 셋 
# <testAR>: test.csv의 예측 날 데이터 셋 
# <flen>: filter length(예측전날의 몇개의 데이터를 가져다 쓸지 결정)
# <test_data>: 실제로 예측 하고 싶은 전날의 데이터(TEST 데이터)    
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------
# [Output]
# <avr_smape>: Training set으로 테스트 했을 때, smape
# <fcst>: test_data의 예측 결과 
# <pred>: Traing set으로 예측했던 예측 결과    
#------------------------------------------------------------------------------        
def linear_prediction(trainAR, testAR, flen, test_data):
    
    len_tr = len(trainAR[0,:])   # 시간 포인트 수 
    day_t = len(trainAR)
    pred = np.empty((len(trainAR),len_tr))
    fcst = np.empty((len(trainAR),len_tr))
    
    for j in range(0, day_t):
        if day_t>1:
            x_ar=np.delete(trainAR[:,len_tr-flen:len_tr], (j), axis=0)
            y=np.delete(testAR, (j), axis=0)
        else:
            x_ar = trainAR[:,len_tr-flen:len_tr]
            y = testAR
            
        pi_x_ar = np.linalg.pinv(x_ar)
        lpc_c = np.empty((len(x_ar),flen))

        
        lpc_c=np.matmul(pi_x_ar, y)
        
        test_e = trainAR[j,:]
        test_ex = test_e[len_tr-flen:len_tr]
        pred[j,:]=np.matmul(test_ex, lpc_c)  
        
    
    x_ar = trainAR[:,len_tr-flen:len_tr]
    y = testAR
    pi_x_ar = np.linalg.pinv(x_ar)
    lpc_c = np.empty((len(x_ar),flen))

        
    lpc_c=np.matmul(pi_x_ar, y)
    
        
    Test_AR = testAR[0:len(testAR),:]
        
    smape_list=np.zeros((len(pred),1))

    for i in range(0,len(pred)):
        smape_list[i]=smape(pred[i,:], Test_AR[i,:])
        avr_smape = np.mean(smape_list)  
    
    test_e = test_data
    test_ex = test_e[len_tr-flen:len_tr]   
    fcst = np.matmul(test_ex,lpc_c)

    return avr_smape, fcst, pred


#%% Local function <Similar day approach module>
# Similar day approaach method 구현.
# 같은 요일 타입의 날의 데이터를 N개를 추출하여 평균을 취하여 사용함.    
#------------------------------------------------------------------------------
# [Input]
# <trainAR>: test.csv의 예측 전날 데이터 셋 
# <testAR>: test.csv의 예측 날 데이터 셋 
# <slen>: 추출한 날의 수 (N)
# <sim_set>: 실제로 예측 하고 싶은 전날의 데이터(TEST 데이터)    
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------
# [Output]
# <simil_smape>: Training set으로 테스트 했을 때, smape
# <simil_temp>: test_data의 예측 결과   
#------------------------------------------------------------------------------
def similar_approach(trainAR, testAR, slen, sim_set):
    simil_smape_list = np.zeros([1,len(testAR[:,0])])
    
    for col_ii in range(0,len(testAR[:,0])):    
        simil_mean = []
        simil_temp =np.zeros([1,24])
        simil_idx =np.zeros([1,len(testAR[:,0])])
        
        for sub_col in range(0,len(testAR[:,0])):
            simil_idx[0,sub_col] = smape(trainAR[col_ii,:],trainAR[sub_col,:])
            
        testAR_temp = np.delete(testAR, np.argmin(simil_idx), axis=0)
        simil_idx = np.delete(simil_idx, np.argmin(simil_idx), axis=1)
        
        for search_len in range(0,slen):
            simil_mean.append(testAR_temp[np.argmin(simil_idx), :])
            testAR_temp = np.delete(testAR, np.argmin(simil_idx), axis=0)
            simil_idx = np.delete(simil_idx, np.argmin(simil_idx), axis=1)
              
        for row_ii in range(0, 24):           
            simil_temp[0, row_ii] = np.median(testAR_temp[:,row_ii])

        simil_smape_list[0,col_ii] = smape(testAR[col_ii,:], simil_temp)
        simil_smape  = np.mean(simil_smape_list)
    
    simil_mean = []
    simil_temp =np.zeros([1,24])
    simil_idx =np.zeros([1,len(testAR[:,0])])
    testAR_temp = testAR
    
    for sub_col in range(0,len(testAR[:,0])):
        simil_idx[0,sub_col] = smape(sim_set,trainAR[sub_col,:])

    
    for search_len in range(0,slen):
        simil_mean.append(testAR_temp[np.argmin(simil_idx),:])
        testAR_temp = np.delete(testAR, np.argmin(simil_idx), axis=0)
        simil_idx = np.delete(simil_idx, np.argmin(simil_idx), axis=1)
   
    for row_ii in range(0, 24):           
        simil_temp[0, row_ii] = np.median(testAR_temp[:,row_ii])
    
    return simil_smape, simil_temp



#%% Local function <Random forest module>
# Random forest를 이용한 regression 기반의 forecasting algorithm
# 전날의 24시간 프로파일을 이용해 다음날 24시간의 프로파일을 예측하는 시스템.   
#------------------------------------------------------------------------------
# [Input]
# <trainAR>: test.csv의 예측 전날 데이터 셋 
# <testAR>: test.csv의 예측 날 데이터 셋 
# <x_24hrs>: 실제로 예측 하고 싶은 전날의 데이터(TEST 데이터)  
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------
# [Output]
# <ypr>: x_24hrs를 이용한 예측 결과  
# <avr_smape>: validation set으로 확인한, smape      
#------------------------------------------------------------------------------
def machine_learn_gen(trainAR, testAR, x_24hrs):
    
    Dnum=trainAR.shape[0]
    lnum=trainAR.shape[1]
    smape_list=np.zeros([Dnum,1])
    
    for ii in range(0,Dnum): # cross validation
        trainAR_temp = np.delete(trainAR, ii, axis=0)
        testAR_temp  = np.delete(testAR, ii, axis=0)
        
        # mae 기반의 loss를 이용한 randomforest model 생성
        regr = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100, criterion='mae')
        regr.fit(trainAR_temp, testAR_temp)
        
        
        x_temp = np.zeros([1,lnum])
        for kk in range(0,lnum):
            x_temp[0,kk] = trainAR[ii, kk]
            
        ypr = regr.predict(x_temp)

        yre = np.zeros([1,lnum])
        for kk in range(0,lnum):
            yre[0,kk] = testAR[ii, kk]
        
        smape_list[ii] = smape(np.transpose(ypr),np.transpose(yre))
        
    regr = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100, criterion='mae')
    regr.fit(trainAR, testAR)
        
    x_24hrs = np.reshape(x_24hrs,(-1,lnum))
    
    avr_smape = np.mean(smape_list)
    ypr=regr.predict(x_24hrs)
    
    return ypr,  avr_smape, smape_list


#%% Local function <DNN module>
# DNN를 이용한 regression 기반의 forecasting algorithm
# 전날의 24시간 프로파일을 이용해 
# 다음날 24시간의 프로파일을 예측할 수 있는 DNN 모델 생성.    
#------------------------------------------------------------------------------
# [Input]
# <trainAR>: test.csv의 예측 전날 데이터 셋 
# <testAR>: test.csv의 예측 날 데이터 셋 
# <EPOCHS>: DNN의 학습을 위한 epoch size  
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------
# [Output]
# <model>: 학습된 모델 출력  
# <avr_smape>: test set으로 테스트 했을 때의 평균, smape  
# <smape_list>: test set으로 테스트 했을 때의 각각, smape      
#------------------------------------------------------------------------------     
def non_linear_model_gen(trainAR, testAR, EPOCHS):
    
    numData=np.size(trainAR,0)
    numTr=int(numData*0.8)
    Xtr=trainAR[0:numTr-1,:]
    Ytr=testAR[0:numTr-1,:]
    
    Xte=trainAR[numTr:numData,:]
    Yte=testAR[numTr:numData,:]
    
    num_tr = np.size(trainAR,1)
    num_te = np.size(testAR,1)
    
    def build_model():
          model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(num_tr,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_te)
          ])
        
          optimizer = tf.keras.optimizers.Adam(0.001)
        
          model.compile(loss='mae',
                        optimizer=optimizer,
                        metrics=['mae', 'mse'])
          return model

    model = build_model()
#    model.summary()
    
    #example_batch = Xtr[:10]
    #example_result = model.predict(example_batch)
    #example_result
    
    class PrintDot(keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
    
    
    history = model.fit(
      Xtr, Ytr,
      epochs=EPOCHS, verbose=0,
      callbacks=[PrintDot()])
   
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    
    Ypr = model.predict(Xte)
    
    smape_list=np.zeros((len(Ypr),1))
    
    for i in range(0,len(Ypr)):
        smape_list[i]=smape(Ypr[i,:], Yte[i,:])
    avr_smape=np.mean(smape_list)
    
    
    return model, avr_smape

#%% ==============================<Main>=======================================
#%% Section #1: Loading data...    
#------------------------------------------------------------------------------
# [Output]
# <test> : test.csv
# <submission>: Data frame for submission.
#------------------------------------------------------------------------------ 
os.chdir('data')                    # Changing Dir. (<main folder>/data)
test = pd.read_csv('test.csv')
submission = pd.read_csv('submission_1002.csv')
os.chdir('..')                      # Changing Dir. (<main folder>)

test['Time'] = pd.to_datetime(test.Time)
test = test.set_index('Time')

print('Section [1]: Loading data...............')
#%% Section #2: Data generation for training set  
#------------------------------------------------------------------------------
# [Input]
# <test> : Test set
# <submission>: Data frame for submission.
# <prev_type>: Day type of the previous day (workday = 1, weekend =2).
# <curr_type>: Day type of the current day (workday = 1, weekend =2).
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------
# [Output]
# <testAR> : Test set
# <trainAR>: Training set
# <subm_24hrs>: test data for 24hrs prediction
#------------------------------------------------------------------------------ 
agg = {}
comp_smape =[]
key_idx = 0

for key in test.columns:
    key_idx = key_idx + 1
    print([key,key_idx])
    prev_type = 2   # 전날 요일 타입
    curr_type = 2   # 예측날 요일 타입
    trainAR, testAR = AR_data_set(test, key, prev_type, curr_type)
    
    print('Section [2]: Data generation for training set...............')

    # [시간 예측을 위한 마지막 24pnt 추출]
    # NaN 값처리를 위해서 마지막 40pnts 추출 한 후에 
    # interpolation하고 나서 24pnts 재추출 
    temp_test = test[key].iloc[8759-40:]      
    temp_test = temp_test.interpolate(method='spline', order=2)
    
    temp_test = np.array(temp_test.values)
    temp_test = temp_test[len(temp_test)-24:len(temp_test)+1]
    subm_24hrs = temp_test
    
    del temp_test
#%% Section #3: Anormaly detection using AR model  
#------------------------------------------------------------------------------
# [Input]
# <testAR> : Test set
# <trainAR>: Training set
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------
# [Output]
# <testAR> : Test set
# <trainAR>: Training set
#------------------------------------------------------------------------------   
    fchk = 1        # filter length
    temp_idx = []
    smape_lin = []
    
    # 한 행씩 linear prediction을 테스트해보고 NaN이 발견된다면, 그 행을 제거.
    for chk_bad in range(0, len(trainAR[:,0])):
        prev_smape = 200 # SMAPE 기준값 
        nan_chk = 0      # NaN chk idx
        
        trainAR_temp = np.zeros([1,24])     # pre-allocation
        testAR_temp = np.zeros([1,24])      # pre-allocation
        
        # 한 행씩 테스트를 하기 위한 변수 설정
        for ii in range(0,24):
            trainAR_temp[0,ii] = trainAR[chk_bad,ii]
            testAR_temp[0,ii] = testAR[chk_bad,ii]
        
        # linear prediction test
        lin_sampe, fcst_temp, pred_hyb = linear_prediction(trainAR_temp, testAR_temp, fchk, subm_24hrs)
            
        if np.isnan(lin_sampe):     # SMAPE가 NaN 경우, 그 행을 제거
            nan_chk = 1
        if np.isnan(np.sum(trainAR_temp)): # chk_bad의 행이 NaN을 포함할 경우 제거
            nan_chk = 1         
        if np.isnan(np.sum(testAR_temp)): # chk_bad의 행이 NaN을 포함할 경우 제거
            nan_chk = 1
                
        if nan_chk == 1: #NaN 값이 있는 행 넘버를 append
            temp_idx.append(chk_bad)
    
    # NaN 값이 나타난 data set은 제거 
    trainAR = np.delete(trainAR, temp_idx, axis=0)
    testAR = np.delete(testAR, temp_idx, axis=0)                 
        
    del_smape = np.zeros([1,len(trainAR[:,1])])
    prev_smape = 200
    fchk = 0
    
    # filter length 최적화 
    for chk in range(3,24):
        # filter length을 바꿔가며 Smape가 최소가 되는 값을 찾아감.
        lin_sampe, fcst_temp, pred_hyb = linear_prediction(trainAR, testAR, chk, subm_24hrs)
        if prev_smape>lin_sampe:
            fchk = chk
            prev_smape = lin_sampe    
   
    # 필요없는 데이터 제거
    # 한 줄(하루)씩 제거해가면서 SMAPE 결과를 분석.
    for chk_lin in range(0,len(trainAR[:,1])):
        
        trainAR_temp = np.delete(trainAR, chk_lin, axis=0)
        testAR_temp = np.delete(testAR, chk_lin, axis=0)
        lin_sampe, fcst_temp, pred_hyb = linear_prediction(trainAR_temp, testAR_temp, fchk, subm_24hrs)          
         
        del_smape[0,chk_lin] = lin_sampe
    
    # SMAPE에 악영향을 주는 행을 제거      
    trainAR = np.delete(trainAR, np.argmin(del_smape), axis=0)
    testAR = np.delete(testAR, np.argmin(del_smape), axis=0)
    del_smape = np.delete(del_smape, np.argmin(del_smape), axis =1)

    print('Section [3]: mitigating bad data...............')
    
    del nan_chk, lin_sampe, fcst_temp, pred_hyb, prev_smape, temp_idx
#%% Section #4: Prediction test
#------------------------------------------------------------------------------
# [Output]
# <fcst> : Predicted hour profile result 
#------------------------------------------------------------------------------ 
    
    # DNN model 
    EPOCHS = 80
    Non_NNmodel, non_smape = non_linear_model_gen(trainAR, testAR, EPOCHS)
    
    # random forest model
    mac_fcst, Mac_smape, smape_listss = machine_learn_gen(trainAR, testAR,subm_24hrs)
    
    # linear model
    lin_sampe, fcst_temp, pred_hyb = linear_prediction(trainAR, testAR, fchk, subm_24hrs)
    
    # Similar day approach model
    temp_24hrs = np.zeros([1,24])    # np.array type으로 변경. 
    for qq in range(0,24):
        temp_24hrs[0,qq] = subm_24hrs[qq]
    
    # Similar day approach model 최적화 (몇 개의 날(N)을 가져오는 게 좋은 지 평가.)    
    prev_smape = 200
    fsim = 0  # N개의 날 
    for sim_len in range(2, 5):    
        sim_smape,  fcst_sim = similar_approach(trainAR, testAR, sim_len, temp_24hrs)
        if prev_smape>sim_smape:
            fsim = sim_len
            prev_smape = sim_smape
            
    # Similar day approach model       
    sim_smape,  fcst_sim = similar_approach(trainAR, testAR, fsim, temp_24hrs)
    # ---------------------------------------------------------------------------------------    
    
    minor_idx = 0 # Autoregression model에서 minor value가 나타나면, 
    # 모델을 Autoregression model에서 similar day appreach로 변경 진행.
    
    # SMAPE가 linear model이 가장 작으면, 해당 결과 사용
    if (lin_sampe<non_smape)&(lin_sampe<Mac_smape)&(lin_sampe<sim_smape):    
        fcst = np.zeros([1,24])  
        for qq in range(0,24):
            fcst[0,qq] = fcst_temp[qq]
            
            if fcst_temp[qq]<0:
                minor_idx = minor_idx+1
                
    # SMAPE가 DNN model이 가장 작으면, 해당 결과 사용
    if (non_smape<lin_sampe)&(non_smape<Mac_smape)&(non_smape<sim_smape):
        temp_24hrs = np.zeros([1,24])
        for qq in range(0,24):
            temp_24hrs[0,qq] = subm_24hrs[qq]
            
        fcst = Non_NNmodel.predict(temp_24hrs)
    
    # SMAPE가 random forest model이 가장 작으면, 해당 결과 사용
    if (Mac_smape<non_smape)&(Mac_smape<lin_sampe)&(Mac_smape<sim_smape):
        fcst = mac_fcst
    
    # SMAPE가 Similar day approach model이 가장 작으면, 해당 결과 사용        
    if (sim_smape<non_smape)&(sim_smape<lin_sampe)&(sim_smape<Mac_smape):
        fcst = fcst_sim
        
    if (minor_idx>0):
        fcst = fcst_sim
    
    # 각 SMAPE 결과 값을 정    
    comp_smape.append([non_smape, lin_sampe,Mac_smape, sim_smape])

    
    a = pd.DataFrame() # a라는 데이터프레임에 예측값을 정리합니다.
    
    print('Section [4]: Hour prediction model...............')    
    for i in range(24):
        a['X2018_7_1_'+str(i+1)+'h']=[fcst[0][i]] # column명을 submission 형태에 맞게 지정합니다.
        

#%% Section #5: Day prediction
#------------------------------------------------------------------------------
# [Output]
# <fcst_d>: Predicted day profile (result)
#------------------------------------------------------------------------------ 
    
    fcst_d = np.zeros([1,10])       # pre-allocation of the result data
    
    
    trainAR_Day = AR_day_set(test, key)  #데이터를 불러옵니다.
    
    # Similar day aprroach
    day_idx = np.zeros([1, 10])
    for ii in range(0, 10):
        mod_idx = -1
        temp_idx =  (ii+mod_idx) % 7     #예측하는 날에 맞는 요일 idx를 정리.
        day_idx[0, ii]  = temp_idx
        
    
    for ii in range(0,10):
        flen = np.random.randint(3)+2    # 2~5개까지 랜덤하게 과거 데이터를 불러옵니다.
    
        temp_day = np.zeros([1, flen])
        
        for jj in range(0, flen):  
            temp_day[0,jj] =  trainAR_Day[jj, int(round(day_idx[0, ii]))]
        
        # 불러온 데이터를 평균을하여 예측함.
        fcst_d[0,ii] = np.mean(temp_day)
    
    print('Section [5]: Day prediction model...............')
    
    for i in range(10):
        a['X2018_7_'+str(i+1)+'_d']=[fcst_d[0][i]] # column명을 submission 형태에 맞게 지정합니다.
    
    del mod_idx, temp_idx
#%% Section #6: Month prediction
#------------------------------------------------------------------------------
# [Output]
# <pred_(N)m>: N번째 달의 전력 사용량 예측 결과
#------------------------------------------------------------------------------ 

    mon_test = np.zeros([1,300])
    
    # Similar day aprroach 
    day_idx = np.zeros([1, 300])
    for ii in range(0, 300):
        mod_idx = -1
        temp_idx = (ii+mod_idx) % 7  # 요일 idx 생성(월~일: 0~6)
        day_idx[0, ii]  = temp_idx
    
    # 휴일의 경우, 일요일과 같은 데이터로 가정함.
    day_idx[0,31+15-1] = 6 # 광복절
    day_idx[0,31+31+24-1] = 6 # 추석
    day_idx[0,31+31+25-1] = 6 # 추석
    day_idx[0,31+31+26-1] = 6 # 대체휴일
    day_idx[0,31+31+30+3-1] = 6 # 개천절
    day_idx[0,31+31+30+9-1] = 6 # 한글날
    day_idx[0,31+31+30+31+30+25-1] = 6 # 성탄절
    
   
    for ii in range(0,300):
        flen = np.random.randint(3)+1  # Similar day approach를 위한 1~4개의 데이터 추출
    
        temp_day = np.zeros([1, flen])
        
        for jj in range(0, flen):  
             temp_day[0,jj] =  trainAR_Day[jj, int(round(day_idx[0, ii]))]
            
        mon_test[0,ii] = np.mean(temp_day)
    
   
    # 결과 합         
    pred_7m = np.sum(mon_test[0,0:31])
    pred_8m = np.sum(mon_test[0,31:62])
    pred_9m = np.sum(mon_test[0,62:92])
    pred_10m = np.sum(mon_test[0,92:123])
    pred_11m = np.sum(mon_test[0,123:153])
    
    a['X2018_7_m'] = [pred_7m] # 7월
    a['X2018_8_m'] = [pred_8m] # 8월
    a['X2018_9_m'] = [pred_9m] # 9월
    a['X2018_10_m'] = [pred_10m] # 10월
    a['X2018_11_m'] = [pred_11m] # 11월 
  
    a['meter_id'] = key 
    agg[key] = a[submission.columns.tolist()]  
    
    print('Section [6]: Month prediction model...............')



#%% Section #6: Write the result data...
# makes the result file
# [File name]: dacon_submmision.csv

os.chdir('data')  # Changing Dir. (<main folder>/data)
output1 = pd.concat(agg, ignore_index=False)
output2 = output1.reset_index().drop(['level_0','level_1'], axis=1)
output2['id'] = output2['meter_id'].str.replace('X','').astype(int)
output2 =  output2.sort_values(by='id', ascending=True).drop(['id'], axis=1).reset_index(drop=True)
output2.to_csv('sub_baseline.csv', index=False)
os.chdir('..')      # Changing Dir. (<main folder>)


print('Section [7]: Saving the result files...............')



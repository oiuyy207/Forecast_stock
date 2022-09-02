import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class StockData_1():
    """
        훈련용 데이터세트와 평가용 데이터세트를 만들기 위한 과정.

        파라미터:
            folderPath : 종목시장의 주가 csv가 저장되어있는 디렉토리 경로.
            indexPath : 종목시장의 인덱스 주 csv파일의 경로.
            timewindowsize : LSTM의 입력으로 들어가는 타임의 사이즈.
                             10일 경우 미래주가를 예측하기 위해 이전 10일 주가를 사용.
            windowsizeForPCC : PCC를 계산하기 위해 사용되는 타임의 사이즈.
                               10일 경우 관계주를 구하기 위해 이전 10일 주가를 사용.
            PositiveStockNumber : 긍정관계주 개수.
            NegativeStockNumber : 부정관계주 개수.
            train_test_rate : 훈련:평가 세트 비율. 0.7일 경우 생성된 데이터 세트중 70%는 훈련용, 30%는 평가용으로 사용.
            batchSize : 세트를 나누는 배치 사이즈.
            date_duration : 에셋별 데이터 길이 조정(default=250)

        과정:
            종목시장의 모든 종목의 시가와 인덱스 시가를 읽어옴.
            minmax스케일러를 각 종목과 인덱스에 적용하고 저장.
            PCC계산, 관계주 계산.
            데이터세트 저장.
               
    """
    def __init__(self,folderPath,indexPath,timewindowsize,windowsizeForPCC,PostiveStockNumber,NegativeStockNumber,train_test_rate,batchSize,date_duration=250):        
               
        if(train_test_rate<=0 or train_test_rate>=1):
            raise ValueError('train_test_rate should be between 0 and 1')        
        self.P=PostiveStockNumber
        self.N=NegativeStockNumber        
        self.T=timewindowsize
        self.Tr=windowsizeForPCC
        self.folderPath = folderPath      
        self.indexPath=indexPath    
        self.batchSize=batchSize
        self.date_duration=date_duration
        
        self.train_test_rate=train_test_rate
        self.scaler=MinMaxScaler(feature_range=(-1,1))  
        self.indexScaler = MinMaxScaler(feature_range=(-1,1))

        self.not_include = pd.DataFrame([],dtype=object)

        self.indexPrice = self.loadIndex()
        self.stockPrice = self.loadCSV()

        self.trainSet,self.testSet=self.make_dataset()

        self.batchNum={}

    
    def getBatch(self,option):
        """
        클래스에 저장된 세트를 y,xp,xn,xi,target으로 나누고 batch생성.

        args:
            option='training' or 'evaluation'

        returns:
            batch 제너레이터
            batch={'y','xp','xn','xi','target'}
        """
        if(option is not 'training' and option is not 'evaluation'):
            raise ValueError('option should be "training" or "evaluation".')

        if(option is 'training'):
            returnSet = self.trainSet
        else:
            returnSet = self.testSet
        
        y=[]
        xp=[]
        xn=[]
        xi=[]
        target=[]

        for d in returnSet:
            y.append(d['target_history'])  
            xp.append(d['pos_history'])       
            xn.append(d['neg_history'])       
            xi.append(d['index_history'])       
            target.append(d['target_price'])

        y=np.reshape(y,(-1,self.T,1))
        xp=np.reshape(xp,(-1,10,self.T,1))
        xn=np.reshape(xn,(-1,10,self.T,1))
        xi=np.reshape(xi,(-1,self.T,1))
        target=np.reshape(target,(-1,1))

        print(f"{option:>14} data : " , y.shape,xp.shape,xn.shape,xi.shape,target.shape)     

        batchNum=int(len(y)/self.batchSize)
        self.batchNum[option]=batchNum

        for i in range(batchNum):
            yield {'y':y[i*self.batchSize:(i+1)*self.batchSize],
                   'xp':xp[i*self.batchSize:(i+1)*self.batchSize],
                   'xn':xn[i*self.batchSize:(i+1)*self.batchSize],
                   'xi':xi[i*self.batchSize:(i+1)*self.batchSize],
                   'target':target[i*self.batchSize:(i+1)*self.batchSize]}

    def loadCSV(self):
        """
        csv파일이 있는 폴더를 입력으로 받아 데이터를 읽어옴.
        """
        csvList=os.listdir(self.folderPath)        
        dataframe = pd.DataFrame([])
        

        for csv in csvList:
            data=pd.read_csv(self.folderPath+'/'+csv,engine='python',encoding = "cp949")
            if(len(data)>self.date_duration):
                data=data[-self.date_duration-1:-1]
                data=data.reset_index()
                data=data['Adj Close']

                dataframe=dataframe.append(data,ignore_index=True)
            else:
                self.not_include = self.not_include.append({"CSV":csv,"Length":len(data)}, ignore_index=True)
        dataT=np.array(dataframe).T
        self.scaler.fit(dataT)
        dataT=self.scaler.transform(dataT)
        dataT=dataT.T
        dataframe=pd.DataFrame(dataT)
        dataframe=dataframe.transpose()
        print('StockPrice shape: ',dataframe.shape)
        return dataframe
    
    def loadIndex(self):
        data=pd.read_csv(self.indexPath,engine='python')
        data=data[-self.date_duration-1:-1]        
        data=data.reset_index()
        data=data.fillna(method='ffill')

        data=np.array(data['Adj Close'])
        data=np.reshape(data,(-1,1))

        data=self.indexScaler.fit_transform(data)

        data=pd.DataFrame(np.squeeze(data))        

        print('IndexPrice shape: ',data.shape)
        return data 


    def make_dataset(self):
        """
        예측모델에 사용되는 입력,타겟 데이터세트.
        입력데이터의 shape는 (목표주식+관계주식+인덱스, 타임윈도우사이즈)
        타겟데이터의 shape는 (1,1)
        """
        maxday=max([self.T,self.Tr])
        dataset=[]

        for i in range(maxday,len(self.stockPrice)):
            print(f'making dataset progress : {i}/{len(self.stockPrice)}',end='\r')
            priceSet=self.stockPrice.loc[i-self.T:i-1]
            targetSet=self.stockPrice.loc[i]
            positiveSet,negativeSet=self.calculate_correlation(self.stockPrice.loc[i-maxday:i-1])
            indexSet = self.indexPrice.loc[i-self.T:i-1]

            for targetNum in priceSet.columns:
                target_history=np.reshape(np.array(priceSet[targetNum]),(self.T,1))
                pos_history=np.reshape(np.array(positiveSet[targetNum].T),(10,self.T,1))
                neg_history=np.reshape(np.array(negativeSet[targetNum].T),(10,self.T,1))
                index_history=np.reshape(np.array(indexSet),(self.T,1))
                target_price=np.reshape(np.array(targetSet[targetNum]),(1,1))

                dataset.append({'target_history':target_history,
                                'pos_history':pos_history,
                                'neg_history':neg_history,
                                'index_history':index_history,
                                'target_price':target_price
                            })

        print('making dataset progress : finished\t')
        
        return dataset[:int(len(dataset)*self.train_test_rate)],dataset[int(len(dataset)*self.train_test_rate):]

    def calculate_correlation(self,priceSet):
        """
        Pearson Correlation Coefficient(PCC)를 계산하고,
        높은순으로 긍정관계주, 낮은순으로 부정관계주를 설정한 개수만큼 생성하여 리스트에 저장한후, 리턴.
        입력은 전체 종목의 타임윈도우간의 주가.

        Returns:
            #모든종목의 관계주. 
            긍정관계주 shape = (종목 수, dataframe(T*P))
            부정관계주 shape = (종목 수, dataframe(T*N))
        """    
        positive=[]
        negative=[] 
        corr=priceSet[-self.Tr:].corr(method='pearson')

        for i in corr.columns:
            tempCorr=corr[i].sort_values(ascending=False)
            index_P=tempCorr[1:self.P+1].index
            index_N=tempCorr[-self.N:].index
            
            priceSet=priceSet[-self.T:]
            posSet=priceSet[index_P]
            negSet=priceSet[index_N]
            posSet.columns=range(self.P)
            negSet.columns=range(self.N)
            
            positive.append(posSet)
            negative.append(negSet)
        return positive,negative

class StockData_2():
    """
        훈련용 데이터세트와 평가용 데이터세트를 만들기 위한 과정.(ver2)

        파라미터:
            folderPath : 종목시장의 주가 csv가 저장되어있는 디렉토리 경로.
            indexdata : 종목시장의 인덱스데이터 모음(INDEX_DATA)
            timewindowsize : LSTM의 입력으로 들어가는 타임의 사이즈.
                             10일 경우 미래주가를 예측하기 위해 이전 10일 주가를 사용.
            windowsizeForPCC : PCC를 계산하기 위해 사용되는 타임의 사이즈. 
                               10일 경우 관계주를 구하기 위해 이전 10일 주가를 사용.
            PositiveStockNumber : 긍정관계주 개수.
            NegativeStockNumber : 부정관계주 개수.
            train_test_rate : 훈련:평가 세트 비율. 0.7일 경우 생성된 데이터 세트중 70%는 훈련용, 30%는 평가용으로 사용.
            batchSize : 세트를 나누는 배치 사이즈.
            date_duration : 에셋별 데이터 길이 조정(default=250)
            h : 예측하는 미래 시점(defalt=1)

        과정:
            종목시장의 모든 종목의 시가와 인덱스 시가를 읽어옴.
            minmax스케일러를 각 종목과 인덱스에 적용하고 저장.
            PCC계산, 관계주 계산.
            데이터세트 저장.
               
    """
    def __init__(self,folderPath,indexdata,timewindowsize,windowsizeForPCC,PostiveStockNumber,NegativeStockNumber,train_test_rate,batchSize,date_duration=250,h=1):        
               
        if(train_test_rate<=0 or train_test_rate>=1):
            raise ValueError('train_test_rate should be between 0 and 1')
        self.H=h   
        self.P=PostiveStockNumber
        self.N=NegativeStockNumber        
        self.T=timewindowsize
        self.Tr=windowsizeForPCC
        self.folderPath = folderPath      
        self.indexData=indexdata    
        self.batchSize=batchSize
        self.date_duration=date_duration
        
        self.train_test_rate=train_test_rate
        self.scaler=MinMaxScaler(feature_range=(-1,1))  
        self.indexScaler = MinMaxScaler(feature_range=(-1,1))

        self.not_include = pd.DataFrame([],dtype=object)

        #self.indexPrice = self.loadIndex()
        self.stockPrice = self.loadCSV()

        self.trainSet,self.testSet=self.make_dataset()

        self.batchNum={}

    
    def getBatch(self,option):
        """
        클래스에 저장된 세트를 y,xp,xn,xi,target으로 나누고 batch생성.

        args:
            option='training' or 'evaluation'

        returns:
            batch 제너레이터
            batch={'y','xp','xn','xi','target'}
        """
        if(option is not 'training' and option is not 'evaluation'):
            raise ValueError('option should be "training" or "evaluation".')

        if(option is 'training'):
            returnSet = self.trainSet
        else:
            returnSet = self.testSet
        
        y=[]
        xp=[]
        xn=[]
        xi=[]
        target=[]

        for d in returnSet:
            y.append(d['target_history'])  
            xp.append(d['pos_history'])       
            xn.append(d['neg_history'])       
            xi.append(d['index_history'])       
            target.append(d['target_price'])

        y=np.reshape(y,(-1,self.T,1))
        xp=np.reshape(xp,(-1,self.P,self.T,1))
        xn=np.reshape(xn,(-1,self.N,self.T,1))
        xi=np.reshape(xi,(-1,9,self.T,1))
        target=np.reshape(target,(-1,1))

        print(f"{option:>14} data : " , y.shape,xp.shape,xn.shape,xi.shape,target.shape)     

        batchNum=int(len(y)/self.batchSize)
        self.batchNum[option]=batchNum

        for i in range(batchNum):
            yield {'y':y[i*self.batchSize:(i+1)*self.batchSize],
                   'xp':xp[i*self.batchSize:(i+1)*self.batchSize],
                   'xn':xn[i*self.batchSize:(i+1)*self.batchSize],
                   'xi':xi[i*self.batchSize:(i+1)*self.batchSize],
                   'target':target[i*self.batchSize:(i+1)*self.batchSize]}

    def loadCSV(self):
        """
        csv파일이 있는 폴더를 입력으로 받아 데이터를 읽어옴.
        """
        csvList=os.listdir(self.folderPath)        
        dataframe = pd.DataFrame([])
        

        for csv in csvList:
            data=pd.read_csv(self.folderPath+'/'+csv,engine='python',encoding = "cp949")
            if(len(data)>self.date_duration):
                data=data[-self.date_duration-1:-1]
                data=data.reset_index()
                data=data['Adj Close']

                dataframe=dataframe.append(data,ignore_index=True)
            else:
                self.not_include = self.not_include.append({"CSV":csv,"Length":len(data)}, ignore_index=True)
        dataT=np.array(dataframe).T
        self.scaler.fit(dataT)
        dataT=self.scaler.transform(dataT)
        dataT=dataT.T
        dataframe=pd.DataFrame(dataT)
        dataframe=dataframe.transpose()
        print('StockPrice shape: ',dataframe.shape)
        return dataframe


    def make_dataset(self):
        """
        예측모델에 사용되는 입력,타겟 데이터세트.
        입력데이터의 shape는 (목표주식+관계주식+인덱스, 타임윈도우사이즈)
        타겟데이터의 shape는 (1,1)
        """
        maxday=max([self.T,self.Tr])
        dataset=[]

        for i in range(maxday,len(self.stockPrice)-self.H):
            print(f'making dataset progress : {i}/{len(self.stockPrice)}',end='\r')
            priceSet=self.stockPrice.loc[i-self.T:i-1]
            targetSet=self.stockPrice.loc[i+self.H]
            positiveSet,negativeSet=self.calculate_correlation(self.stockPrice.loc[i-maxday:i-1])
            indexSet = self.indexData.loc[i-self.T:i-1].drop(columns="Date")

            for targetNum in priceSet.columns:
                target_history=np.reshape(np.array(priceSet[targetNum]),(self.T,1))
                pos_history=np.reshape(np.array(positiveSet[targetNum].T),(self.P,self.T,1))
                neg_history=np.reshape(np.array(negativeSet[targetNum].T),(self.N,self.T,1))
                index_history=np.reshape(np.array(indexSet),(9,self.T,1))#???요거 맞나...?아마 9열 맞을껄....?
                target_price=np.reshape(np.array(targetSet[targetNum]),(1,1))

                dataset.append({'target_history':target_history,
                                'pos_history':pos_history,
                                'neg_history':neg_history,
                                'index_history':index_history,
                                'target_price':target_price
                            })

        print('making dataset progress : finished\t')
        
        return dataset[:int(len(dataset)*self.train_test_rate)],dataset[int(len(dataset)*self.train_test_rate):]

    def calculate_correlation(self,priceSet):
        """
        Pearson Correlation Coefficient(PCC)를 계산하고,
        높은순으로 긍정관계주, 낮은순으로 부정관계주를 설정한 개수만큼 생성하여 리스트에 저장한후, 리턴.
        입력은 전체 종목의 타임윈도우간의 주가.

        Returns:
            #모든종목의 관계주. 
            긍정관계주 shape = (종목 수, dataframe(T*P))
            부정관계주 shape = (종목 수, dataframe(T*N))
        """    
        positive=[]
        negative=[] 
        corr=priceSet[-self.Tr:].corr(method='pearson')

        for i in corr.columns:
            tempCorr=corr[i].sort_values(ascending=False)
            index_P=tempCorr[1:self.P+1].index
            index_N=tempCorr[-self.N:].index
            
            priceSet=priceSet[-self.T:]
            posSet=priceSet[index_P]
            negSet=priceSet[index_N]
            posSet.columns=range(self.P)
            negSet.columns=range(self.N)
            
            positive.append(posSet)
            negative.append(negSet)
        return positive,negative

class StockData_M6():
    """
        훈련용 데이터세트와 평가용 데이터세트를 만들기 위한 과정.
        M6 compitition용(22-05-19)

        >> Ver.1 <<

        파라미터:
            asset_path : 종목시장의 주가 csv가 저장되어있는 디렉토리 경로.
            index_data : 인덱스 데이터.
            time_window_size : LSTM의 입력으로 들어가는 타임의 사이즈. 10일 경우 미래주가를 예측하기 위해 이전 10일 주가를 사용.
            train_test_rate : 훈련:평가 세트 비율. 0.7일 경우 생성된 데이터 세트중 70%는 훈련용, 30%는 평가용으로 사용.
            batchSize : 세트를 나누는 배치 사이즈.

        과정:
            종목시장의 모든 종목의 시가읽어옴.
            각 종목과 Date기준으로 인덱스와 결합.
            minmax스케일러를 각 종목과 인덱스에 적용하고 저장.
            데이터세트 저장.
               
    """
    def __init__(self,asset_path,index_data,time_window_size,train_test_rate,batchSize):        
               
        if(train_test_rate<=0 or train_test_rate>=1):
            raise ValueError('train_test_rate should be between 0 and 1')          
        self.T=time_window_size

        self.asset_path = asset_path
        self.index_data=index_data

        self.batchSize=batchSize
        self.train_test_rate=train_test_rate

        self.scaler=MinMaxScaler(feature_range=(-1,1))
        self.scaler_min_max_values = pd.DataFrame([],dtype=object)
        self.index_scaler = MinMaxScaler(feature_range=(-1,1))
        self.index_scaler_min_max_values = pd.DataFrame([],dtype=object)

        self.all_assets_dataframe_list = self.make_data_list()

        self.trainSet,self.testSet=self.make_dataset()

        self.batchNum={}

    
    def getBatch(self,option):
        """
        클래스에 저장된 세트를 y,xi,target으로 나누고 batch생성.

        args:
            option='training' or 'evaluation'

        returns:
            batch 제너레이터
            batch={'y','xi','target'}
        """
        if(option is not 'training' and option is not 'evaluation'):
            raise ValueError('option should be "training" or "evaluation".')

        if(option is 'training'):
            returnSet = self.trainSet
        else:
            returnSet = self.testSet
        
        y=[]
        xi=[]
        target=[]
        #asset_stock / index_data / target
        for d in returnSet:
            y.append(d['asset_stock'])  
            xi.append(d['index_data'])       
            target.append(d['target'])

        y=np.reshape(y,(-1,self.T,1))
        xi=np.reshape(xi,(-1,9,self.T,1))
        target=np.reshape(target,(-1,1))

        print(f"{option:>14} data : " , y.shape,xi.shape,target.shape)

        batchNum=int(len(y)/self.batchSize)
        self.batchNum[option]=batchNum

        for i in range(batchNum):
            yield {'y':y[i*self.batchSize:(i+1)*self.batchSize],
                   'xi':xi[i*self.batchSize:(i+1)*self.batchSize],
                   'target':target[i*self.batchSize:(i+1)*self.batchSize]}

    def make_data_list(self):
        """
        csv파일이 있는 폴더를 입력으로 받아 데이터를 읽어옴.
        각 에셋별로 전부 스케일링함.
        args:

        returns:
            asset_data_list : 각 에셋별 스케일링(-1 ~ +1)된 pd.DataFrame(에셋의 Adj close + INDEX_DATA)을 원소로 가지는 list.
            self.scaler_min_max_values와
            self.index_scaler_min_max_values를 채워 넣음.
        """
        csv_list=os.listdir(self.asset_path)
        
        asset_data_list = []#실제 return
        col_list = []#self.scaler_min_max_values의 columns용
        
        for csv in csv_list:
            data=pd.read_csv(self.asset_path+'/'+csv,engine='python',encoding = "cp949")

            data = data[['Date','Adj Close']]
            data = data.dropna(axis=0)
            data_table = pd.merge(left = data, right = self.index_data, on='Date', how='left')
            data_table = data_table.fillna(method='ffill')
            #data_table = data_table.fillna(method='bfill')
            data_table = data_table.drop(columns="Date")#"Date"열 제거
            self.scaler.fit(data_table)
            data_table = self.scaler.transform(data_table)
            data_table = pd.DataFrame(data_table)

            t=pd.DataFrame([self.scaler.data_min_, 
                            self.scaler.data_max_])
            self.scaler_min_max_values = pd.concat([self.scaler_min_max_values, t[0]],
                                                   axis = 1,
                                                   ignore_index=True)
            col_list.append(csv[:-4])

            asset_data_list.append(data_table)
        
        self.scaler_min_max_values.columns = col_list
        self.index_scaler_min_max_values = t.iloc[:,1:]
        self.index_scaler_min_max_values.columns = self.index_data.columns[1:]

        return asset_data_list

    def make_dataset(self):
        """
        예측모델에 사용되는 입력,타겟 데이터세트.
        asset_stock의 shape는 (-1,1)
        index_data의 shape는 (2,-1,1)
        target의 shape는 (1,1)

        args:

        returns:
            trainset, testset
        """
        maxday=self.T
        dataset=[]
        #self.all_assets_dataframe_list
        for i in tqdm(range(100),desc = "making dataset progress",ncols=47):

            asset_df = self.all_assets_dataframe_list[i].copy()
            for j in range(maxday,len(asset_df)-20):
                price_set_T = asset_df.loc[j-self.T:j-1].copy().to_numpy()
                target_T=asset_df.iloc[j+20,0].copy()
                
                #reshape!
                asset_stock = price_set_T[:,0]
                index_data = price_set_T[:,1:]

                asset_stock = asset_stock.reshape((-1,1))
                index_data = index_data.T.reshape((9,-1,1))
                target_T = target_T.reshape((1,1))

                dataset.append({'asset_stock':asset_stock,
                                'index_data':index_data,
                                'target':target_T
                            })
        
        train_test_split_idx = int(len(dataset)*self.train_test_rate)

        return dataset[:train_test_split_idx],dataset[train_test_split_idx:]




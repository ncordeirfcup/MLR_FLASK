from sklearn import metrics
import numpy as np
import math
from sklearn.model_selection import KFold  #For K-fold cross validation\n
import random
import copy
import math
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
from rm2 import rm2
from sklearn.metrics import r2_score
from scipy import stats
from sklearn.model_selection import cross_val_predict

class cross_validation:
      def __init__(self,X_data,y_data, n, model,cv):
            self.n=n
            self.X_data=X_data
            self.y_data=y_data
            self.cv=cv
            self.model=model
      
      def aard(self,df):
          #df['Active2']=np.exp(df['Active'])
          #df['Predict2']= np.exp(df['Predict'])
          df['diff']=abs(df['Active']-df['Predict'])
          aard=(100*(df['diff']/df['Active'])).sum()/(df.shape[0])
          return aard
        
      def fit(self):
         ls2=[]
         cv2 = KFold(n_splits=self.cv, shuffle=True, random_state=42)
         print(self.X_data)
         ypr = cross_val_predict(self.model, pd.DataFrame(self.X_data), self.y_data, cv=cv2)
         a1=pd.concat([self.n, self.y_data, pd.DataFrame(self.model.predict(self.X_data))], axis=1)
         a1.columns=[self.n.columns[0],'Active','Predict']
         a=pd.concat([self.n, self.y_data, pd.DataFrame(ypr)], axis=1)
         a.columns=[self.n.columns[0],'Active','Predict']
         print(a.shape)
         aardcv=self.aard(a[['Active','Predict']])
         a['diff']=abs(a['Active']-a['Predict'])
         #aard=(100*(a['diff']/a['Active'])).sum()/(self.y_data.shape[0])
         r2=(stats.pearsonr(a1['Active'],a1['Predict'])[0])**2
         mae=mean_absolute_error(a['Active'],a['Predict'])
         a['Del_Res']=a['Active']-a['Predict']
         a['Mean']=a['Active'].mean()
         a['nsum']=a['Active']-a['Mean']
         q2lmo=1-((a['Del_Res']**2).sum()/(a['nsum']**2).sum())
         rm2tr,drm2tr=rm2(a.iloc[:,1:2],a.iloc[:,2:3]).fit()
         return r2,mae,q2lmo,rm2tr,drm2tr,a,aardcv
         
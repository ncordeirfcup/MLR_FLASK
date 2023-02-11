import numpy as np
import pandas as pd
class model_params():
    def __init__(self,X,y,model):
        self.X=X
        self.y=y
        self.model=model

    def fit(self):
       cf=self.model.coef_
       ic=self.model.intercept_
       ls,lt,lm=[ic[0]],['Constant'],[]
       for i in cf:
          for j in range(len(i)):
             x=i[j]
             ls.append(round(x,3))
             lt.append(self.X.columns[j])
       N = len(self.X)
       p = len(self.X.columns) + 1  # plus one because LinearRegression adds an intercept term

       X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
       X_with_intercept[:, 0] = 1
       X_with_intercept[:, 1:p] = self.X.values

       beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ self.y.values
       y_hat = self.model.predict(self.X)
       residuals = self.y.values - y_hat
       residual_sum_of_squares = residuals.T @ residuals
       sigma_squared_hat = residual_sum_of_squares[0, 0] / (N - p)
       var_beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * sigma_squared_hat
       for p_ in range(p):
          standard_error = var_beta_hat[p_, p_] ** 0.5
          #print(f"SE(beta_hat[{p_}]): {standard_error}")
          lm.append(round(standard_error,3))
       dc={'Desc': lt, 'Coef':ls, 'Std err':lm}
       print(dc)
       line=''
       for a,b,c in zip(dc['Desc'][1:],dc['Coef'][1:],dc['Std err'][1:]):
           line=line+str(round(b,3))+'(+/-'+str(round(c,3))+')'+'*'+a+'+'
       for b,c in zip(dc['Coef'][0:1],dc['Std err'][0:1]):
           line2=line+str(round(b,3))+'(+/-'+str(round(c,3))+')'
       line3=str(self.y.columns[0])+'='+line2
       result=pd.DataFrame(dc)
       return result,line3

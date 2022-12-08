from flask import Flask, render_template, request, Response
import pandas as pd
import csv
from loo import loo
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from rm2 import rm2
from applicability import apdom
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from model_params import model_params
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power
from sklearn.preprocessing import StandardScaler
import numpy as np


app=Flask(__name__)

app.config['SECRET_KEY']='AmitMLR'

reg=LinearRegression()

def corrl(df):
    lt=[]
    df1=df.iloc[:,0:]
    for i in range(len(df1)):
        x=df1.values[i]
        x = sorted(x)[0:-1]
        lt.append(x)
    flat_list = [item for sublist in lt for item in sublist]
    return max(flat_list),min(flat_list)


def create_figure(yobs, ypred,yobsts, ypredts):
    fig = Figure(figsize=(15,10))
    axis=fig.add_subplot(1, 1, 1)
    axis.plot([yobs.min(), yobs.max()], [yobs.min(), yobs.max()], 'k--', lw=4)
    axis.scatter(yobs, ypred,label='Train', color='blue')
    axis.scatter(yobsts, ypredts,label='Test', color='red')
    axis.set_ylabel('Predicted values',fontsize=28)
    axis.set_xlabel('Observed values',fontsize=28)
    axis.legend(fontsize=18)
    axis.tick_params(labelsize=18)
    return fig
def create_figure2(x,y):
    fig = plt.figure(figsize = (20, 10))
    plt.bar(x,y,align='center') # A bar chart
    plt.xlabel('Descriptors', fontsize = 20)
    plt.ylabel('Standardized coefficients', fontsize = 20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    return fig

def create_figure3(df,dfts,Xtr,Xts,ytr,ypr,yts,ytspr):
    scaler_res = StandardScaler()
    st_residuals_train = scaler_res.fit_transform(df.iloc[:,-1:])
    st_residuals_val = scaler_res.transform(dfts.iloc[:,-1:])
    h_k = (3 * int(len(Xtr.columns)+1))/len(Xtr.index)
    res_tr=pd.DataFrame(st_residuals_train, columns=['Standardized Residuals'])
    res_ts=pd.DataFrame(st_residuals_val, columns=['Standardized Residuals'])


    inner = matrix_power((np.dot(Xtr.T,Xtr)), -1)
    h_indexes = []
    for idx in Xtr.index.values:
        xi_t = np.asarray(Xtr.loc[idx]).T
        xi = np.asarray(Xtr.loc[idx])
        hi = np.dot(xi_t.dot(inner), xi)
        h_indexes.append(hi)

    h_indexes_v = []
    for idx in Xts.index.values:
        xi_t = np.asarray(Xts.loc[idx]).T
        xi = np.asarray(Xts.loc[idx])
        hi_v = np.dot(xi_t.dot(inner), xi)
        h_indexes_v.append(hi_v)

    htr=pd.DataFrame(h_indexes, columns=['Leverage'])
    hts=pd.DataFrame(h_indexes_v, columns=['Leverage'])
    global ftr
    ftr=pd.concat([df,res_tr,htr], axis=1)
    global fts
    fts=pd.concat([dfts,res_ts,hts], axis=1)

    fig=plt.figure(figsize=(15,10))
    plt.scatter(h_indexes, st_residuals_train, s=120, marker='s', edgecolors='k', c="tab:blue",alpha=0.6)
    plt.scatter(h_indexes_v, st_residuals_val, s=120, marker='o', edgecolors='k',c="tab:red", alpha=0.6)

    plt.ylim(-4, 4)
    plt.xlim(right=1)
    plt.xlim(left=0)

    leg = list()
    leg.append('Train')
    leg.append('Test')

    plt.legend(leg, loc='best', frameon=True, fontsize=16)
    plt.xlabel('Leverages', fontsize=20)
    plt.ylabel('Standarized residuals', fontsize=20)
    plt.axhline(y=3, color='gray', linestyle='--')
    plt.axhline(y=-3, color='gray', linestyle='--')
    plt.axvline(x=h_k, linestyle='--')
    plt.tight_layout()
    #plt.show()
    #plt.savefig('williams_ad.jpg', dpi=300)
    return fig

def r2pr(Xts,yts,model,m):
    ytspr=pd.DataFrame(model.predict(Xts))
    ytspr.columns=['Pred']
    rm2ts,drm2ts=rm2(yts,ytspr).fit()
    tsdf=pd.concat([yts,pd.DataFrame(ytspr)],axis=1)
    tsdf.columns=['Active','Predict']
    tsdf['Aver']=m
    tsdf['Aver2']=tsdf['Predict'].mean()
    tsdf['diff']=tsdf['Active']-tsdf['Predict']
    tsdf['diff2']=tsdf['Active']-tsdf['Aver']
    tsdf['diff3']=tsdf['Active']-tsdf['Aver2']
    r2pr=1-((tsdf['diff']**2).sum()/(tsdf['diff2']**2).sum())
    r2pr2=1-((tsdf['diff']**2).sum()/(tsdf['diff3']**2).sum())
    RMSEP=((tsdf['diff']**2).sum()/tsdf.shape[0])**0.5
    return ytspr,r2pr,r2pr2,RMSEP,rm2ts,drm2ts

def model(Xtr,Xts,ytr,yts,file1):
    r2=reg.score(Xtr,ytr)
    ypr=pd.DataFrame(reg.predict(Xtr), columns=['Pred'])
    cv=loo(Xtr,ytr,file1)
    c,m,l=cv.cal()
    ypr=pd.DataFrame(reg.predict(Xtr), columns=['Pred'])
    rm2tr,drm2tr=rm2(ytr,l).fit()
    d=mean_absolute_error(ytr,ypr)
    e=(mean_squared_error(ytr,ypr))**0.5
    return r2,c,d,e,rm2tr,drm2tr,m,l,ypr


@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')


@app.route('/data', methods=['GET','POST'])
def data():
    if request.method == 'POST':
       file_tr = request.form['csvfile_tr']
       data_tr = pd.read_csv(file_tr)
       file_ts = request.form['csvfile_ts']
       data_ts = pd.read_csv(file_ts)
       ntr=data_tr.iloc[:,0:1]
       nts=data_ts.iloc[:,0:1]
       if request.form['options']=='first':
          Xtr=data_tr.iloc[:,2:]
          ytr=data_tr.iloc[:,1:2]
       elif request.form['options']=='last':
          Xtr=data_tr.iloc[:,1:-1]
          ytr=data_tr.iloc[:,-1:]
       Xts=data_ts[Xtr.columns]
       yts=data_ts[ytr.columns]
       global dc
       dc=Xtr.corr()
       mx,mn=corrl(dc)
       mc=max(abs(mx),abs(mn))
       reg.fit(Xtr,ytr)
       mp=model_params(Xtr,ytr,reg)
       tbl,eq=mp.fit()
       tbl1=tbl[1:].sort_values('Coef', ascending=False)
       y=tuple(list(tbl1['Coef']))
       x=tuple(list(tbl1['Desc']))
       global figure2
       figure2=create_figure2(x,y)
       r2,q2loo,mae,mse,rm2tr,drm2tr,m,l,ypr = model(Xtr,Xts,ytr,yts,data_tr)
       r2adj=1-((1-r2)*(Xtr.shape[0]-1)/(Xtr.shape[0]-Xtr.shape[1]-1))
       ytspr,r2pred,r2pred2,RMSEP,rm2ts,drm2ts=r2pr(Xts,yts,reg,m)
       adstr=apdom(Xtr,Xtr)
       yadstr=adstr.fit()
       #global df
       df=pd.concat([ntr,Xtr,ytr,ypr,l,yadstr],axis=1)
       df['Residual']=df[ytr.columns[0]]-df['Pred']
       adsts=apdom(Xts,Xtr)
       yadsts=adsts.fit()
       #global dfts
       dfts=pd.concat([nts,Xts,yts,ytspr,yadsts],axis=1)
       dfts['Residual']=dfts[ytr.columns[0]]-dfts['Pred']
       global figure1
       figure1=create_figure(ytr,ypr,yts,ytspr)
       global figure3
       figure3=create_figure3(df,dfts,Xtr,Xts,ytr,ypr,yts,ytspr)
       return render_template('data.html', r2=r2,r2adj=r2adj, q2loo=q2loo,mae=mae,mse=mse,
              rm2tr=rm2tr,drm2tr=drm2tr,r2pred=r2pred, r2pred2=r2pred2,
              rmsep=RMSEP, rm2ts=rm2ts, drm2ts=drm2ts,trsize=ntr.shape[0],
              tssize=nts.shape[0], mc=mc,tbl=tbl.to_html(),eq=eq)

@app.route('/resultsTR')
def results_tr():
    #return redirect(url_for("data", category=category, _external=True, _scheme='https'))
    return render_template('results.html', result =ftr.to_html(index=False))

@app.route('/resultsTS')
def results_ts():
    #return redirect(url_for("data", category=category, _external=True, _scheme='https'))
    return render_template('results.html', result =fts.to_html(index=False))

@app.route('/correlmatrix')
def correlmatrix():
    #return redirect(url_for("data", category=category, _external=True, _scheme='https'))
    return render_template('results.html', result =dc.to_html())

@app.route('/plot.png')
def plot_png():
    output = io.BytesIO()
    FigureCanvas(figure1).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot2.png')
def plot_png2():
    output = io.BytesIO()
    FigureCanvas(figure2).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot3.png')
def plot_png3():
    output = io.BytesIO()
    FigureCanvas(figure3).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


if __name__=='__main__':
  app.run(debug=True)

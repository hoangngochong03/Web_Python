from flask import Blueprint,render_template,session,make_response,request,redirect,url_for
from flask_redis import FlaskRedis
import numpy as np  
import tempfile
import pandas as pd 
import matplotlib
from sklearn.preprocessing import LabelEncoder
matplotlib.use('Agg') 
import io
import os
import base64
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from collections import Counter
from pandas.api.types import is_numeric_dtype
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
views=Blueprint(__name__,"Views")
views.secret_key="21522106"
@views.route("/home")
def home():
    return render_template("index.html") #show file html

# Upload file CSV từ người dùng
@views.route('/process_csv', methods=['POST','GET'])

def process_csv():
        if request.method=="POST":
            csv_file = request.files['file']
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                csv_file.save(tmp.name)
                csv_path = tmp.name
            session['data'] = csv_path
            return redirect(url_for('Views.Show_Data'))
        else:
            return render_template('read_csv.html')
#Show dữ liệu excel ra
@views.route('/Show_Data',methods=['POST','GET'])
def Show_Data():
        data=session.get('data')
        if os.path.isfile(data):
            data=pd.read_csv(data)
        else:
            data=pd.read_csv(io.StringIO(data))
        X=data.isna().sum()

        data_path=data.head(20)
        lst=list(data_path.columns)
        des=data.describe().to_dict()
        return render_template("show_data.html", data_path=data_path.to_dict('records'),statistic=X.to_dict(),columns=lst,des=des)
@views.route('/Statistic',methods=['POST','GET'])
def preprocessing():
    data=session.get('data')
    
    if os.path.isfile(data):
        data=pd.read_csv(data)
    else:
        data=pd.read_csv(io.StringIO(data))

    if request.method=="POST":   
        balance=request.form.get('Predict') 

    lst=list(data.columns)
    tmp=data.copy()
    lst_object=[]
    change=0

    for x in tmp.columns:
        if is_numeric_dtype(tmp[x]):
            continue
        else:
            label_encoder = LabelEncoder()
            label_encoder.fit(tmp[x])
            tmp[x]=label_encoder.transform(tmp[x])
            lst_object.append(dict(zip(data[x].unique(),tmp[x].unique())))
            change=1
    X=data.isna().sum()
    Target_Data=dict(Counter(data[balance]))
    data_path=data.head(20)
    tmp=X.to_dict()
    flag=0
    for i in tmp.values():
         if i!=0:
              flag+=1
    
    return render_template("PreProcessing_data.html",data_path=data_path.to_dict('records'),statistic=X.to_dict(),sta_target=Target_Data,columns=lst,flag=flag,change=change,lst_object=lst_object)

@views.route('/Preprocessing',methods=['POST','GET'])
def missing_values():
    data=session.get('data')
    flag=0
    if not os.path.isfile(data):
        data=pd.read_csv(io.StringIO(data))
    else:
        data=pd.read_csv(data)
        col=data.columns
        if request.method=="POST":
            method=request.form.get('Method')
            if method=="dropna()":
                data=data.dropna()
            elif method==None:
                flag=1
            else:
                imputer=SimpleImputer(strategy=method)
                data=imputer.fit_transform(data)
                data=pd.DataFrame(data,columns=col)
                for column in data.columns:
                    try:
                        data[column] = data[column].astype(float)
                    except ValueError:
                        pass
        
    

    lst=list(data.columns)
    ###
    temp=data.copy()
    lst_object=[]
    change=0

    for x in temp.columns:
        if is_numeric_dtype(temp[x]):
            continue
        else:
            label_encoder = LabelEncoder()
            label_encoder.fit(temp[x])
            temp[x]=label_encoder.transform(temp[x])
            lst_object.append(dict(zip(data[x].unique(),temp[x].unique())))
            change=1
    X=data.isna().sum()
    data_path=data.head(20)
    session.pop('data',None)
    session['data'] = data.to_csv(index=False)#  data dạng chuỗi
    return render_template("resolve_missing.html",data_path=data_path.to_dict('records'),statistic=X.to_dict(),columns=lst,change=change,lst_object=lst_object)
@views.route('/Predict',methods=['POST','GET'])
def Predict():
    reprocess_data=session.get('data')
    if os.path.isfile(reprocess_data):
        clean_data=pd.read_csv(reprocess_data)
    else:
        clean_data=pd.read_csv(io.StringIO(reprocess_data)) # nếu data dạng chuỗi
    if request.method=="POST":   
        Type=request.form.get('type')
        x_data=request.form.get('x')
        balance=request.form.get('Predict') 
        Train_data = request.form.get('Train')
    x_vals = [float(x) for x in x_data.split(' ')]
    x_vals=np.array(x_vals).reshape(1,-1)
    Train_data=[x for x in Train_data.split(' ')]
    tmp=clean_data.copy()
    for x in Train_data:
        if is_numeric_dtype(tmp[x]):
            continue
        else:
            label_encoder = LabelEncoder()
            label_encoder.fit(tmp[x])
            tmp[x]=label_encoder.transform(tmp[x])
    
    input_data = tmp[Train_data]
    target_data = tmp[balance]
    X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.3, random_state=42)
    if Type=='LinearRegression':
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        pred=model.predict(x_vals)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return render_template("predict.html",pred=pred,accuracy=mse,f1=r2,Type=Type,predict=balance)
    elif Type=='LogisticRegression':
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        pred=model.predict(x_vals)
        accuracy = accuracy_score(y_test, y_pred)
        return render_template("predict.html",pred=pred,accuracy=accuracy,Type=Type,predict=balance)
    elif Type=="KNeighborsClassifier":
        k=len(tmp[balance].unique())
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred=model.predict(X_test)
        pred=model.predict(x_vals)
        accuracy = accuracy_score(y_test, y_pred)
        return render_template("predict.html",pred=pred,accuracy=accuracy,Type=Type,predict=balance)
    else:
        return redirect(url_for('Views.Show_Data'))
@views.route('/Draw',methods=['POST','GET'])
def Draw():
        data=session.get('data')
        if os.path.isfile(data):
            data=pd.read_csv(data)
        else:
            data=pd.read_csv(io.StringIO(data)) # nếu data dạng chuỗi
        if request.method=="POST":   
            chart_type=request.form.get('type')
            x_col = request.form.get('X')
            x_data = data[x_col].values
            y_col = request.form.get('Y')
            if y_col!=None:
                y_data = data[y_col].values  
            #VẼ BIỂU ĐỒ DỰA VÀO VALUES CỦA X VÀ Y
            if chart_type == 'line':
                plt.clf()
                plt.figure(figsize=(15,5))
                plt.plot(x_data, y_data)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                img_str = base64.b64encode(img.getvalue()).decode()
                img = f'<img src="data:image/png;base64,{img_str}">'
                return render_template('show_chart.html', img_tag=img) 
                
            elif chart_type == 'scatter':
                plt.clf()
                plt.figure(figsize=(15,5))
                plt.scatter(x_data, y_data)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                # lưu biểu đồ được tạo bởi plt dưới dạng một file hình ảnh PNG rồi mã hóa dưới dạng base64
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0) #đặt con trỏ ở đầu dòng để đọc dữ liệu
                img_str = base64.b64encode(img.getvalue()).decode()
                img = f'<img src="data:image/png;base64,{img_str}">'
                return render_template('show_chart.html', img_tag=img)                
            elif chart_type == 'histogram':
                
                plt.clf()
                plt.figure(figsize=(15,5))
                plt.hist(x_data)
                plt.xlabel(x_col)
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                img_str = base64.b64encode(img.getvalue()).decode()
                img = f'<img src="data:image/png;base64,{img_str}">'
                return render_template('show_chart.html',img_tag=img)
            elif chart_type =="bar":
                plt.clf() # xóa đi cái biểu đồ đã tồn tại trước đó
                plt.figure(figsize=(20,8))
                plt.bar(x_data, y_data, color = 'g', width = 0.72, label = y_col)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.legend()
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                img_str = base64.b64encode(img.getvalue()).decode()
                img = f'<img src="data:image/png;base64,{img_str}">'
                return render_template('show_chart.html',img_tag=img)

        return render_template("show_data.html",data_path=data.to_dict('records'))

from flask import Flask,session # import thư viện flask để tạo web
from Views import views 
from flask_session import Session
app = Flask(__name__, template_folder=r"C:\Users\hongp\OneDrive\Desktop\Do An\DoAnKTLTPT") #khởi tạo ứng dụng và đặt đường dẫn cho template

app.secret_key = '21522106'

app.register_blueprint(views,url_prefix="/views") 
if __name__=='__main__':
    app.run(debug=True,port=8000)

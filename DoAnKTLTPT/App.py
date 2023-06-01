from flask import Flask,session # import thư viện flask để tạo web
from Views import views 
app = Flask(__name__, template_folder=r"C:\Users\hongp\OneDrive\Desktop\Do An\DoAnKTLTPT") #khởi tạo ứng dụng và đặt đường dẫn cho template

app.secret_key = '21522106'
app.register_blueprint(views,url_prefix="/views") 
if __name__=='__main__':
    app.run(debug=True,port=8000) # khi thay đổi các file nó sẽ tự động cập nhật port là 8000 mặc định 5000

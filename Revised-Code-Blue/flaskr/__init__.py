from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.utils import redirect,secure_filename
from flask_uploads import UploadSet,configure_uploads,IMAGES,DATA,ALL
from wtforms import form 
from flaskr.forms import SignupForm
from flask import Flask, render_template, Response, jsonify, url_for, redirect, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import pickle
import os
from flask import Flask, flash, request
import time
from flask_wtf import FlaskForm
from wtforms import StringField,TextAreaField,SubmitField,PasswordField,DateField,SelectField
from wtforms import validators
from wtforms.validators import DataRequired,Email,EqualTo,Length
from flask import Flask, render_template, Response, jsonify, url_for
from tensorflow.keras.preprocessing import image
from flask import Flask, flash, request, redirect, url_for
from flask import session
from flask import Flask
from flask import url_for
from flask import redirect
from flask import request
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from flask_login import current_user, login_user
from flask_login import login_required, current_user, login_user, logout_user
from werkzeug.datastructures import  FileStorage
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import librosa
import librosa.display


class LoginForm(FlaskForm):
    email = StringField("Enter your email", validators=[DataRequired(),Email()])
    password = PasswordField("Password", validators=[DataRequired(),Length(min=6,max=16)])
    submit = SubmitField("Log In")

class SignupForm(FlaskForm):
    email = StringField("Enter your email", validators=[DataRequired(),Email()])
    password = PasswordField("Password", validators=[DataRequired(),Length(min=6,max=16)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(),EqualTo(password)])
    emergency_contact = SelectField('Choose your preferred method of emergency contact when facial paralysis is detected', validators=[DataRequired()], choices=[('Call 911', 'Text Emergency Contact')] )
    submit = SubmitField("Sign Up")

gpus = tf.config.list_physical_devices("GPU")
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

# Connect camera to the application
camera = cv2.VideoCapture(0)

def before_request():
        with app.app_context():
            db.create_all()

app = Flask(__name__)
app.before_request(before_request)
app.config['SECRET_KEY']='thisisfirstflaskapp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','wav','mp3'}
UPLOAD_FOLDER = '/tmp'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user-table.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

db = SQLAlchemy()
 
class UserModel(UserMixin, db.Model):
    __tablename__ = 'users'
 
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(80), unique=True)
    username = db.Column(db.String(100))
    password_hash = db.Column(db.String())
    emergency_contact = db.Column(db.String(80), unique=False)
 
    def set_password(self,password):
        self.password_hash = generate_password_hash(password)
     
    def check_password(self,password):
        return check_password_hash(self.password_hash,password)

#Create form class
from flask_login import LoginManager
login = LoginManager()
 
@login.user_loader
def load_user(id):
    return UserModel.query.get(int(id))

db.init_app(app)
before_request()
def create_table():
    db.create_all()
login.init_app(app) # type: ignore
login.login_view = 'login'

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

"""@app.route('/uploader', methods = ['GET','POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)"""

"""@app.route('/upload-predict', methods=['GET','POST']) #so upload-predict takes POST then because of upload.html
def upload_image_button():       
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        predict_asymmetry_upload(cv2.imread(path)) #calls method below which returns the appropriate URL. 
        #if it says it returns nothing...add return in front of the line above
        #print('upload_image filename: ' + filename)
        #flash('Image successfully uploaded and displayed below')
        #return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)"""
def predict_asymmetry_upload():
    """ Face detection section """
    # Load stroke detection model
    classifier = load_model("/Users/erickacorral/Desktop/Revised-Code-Blue/Code-Blue/Revised-Code-Blue/flaskr/StrokeDetectionModel.h5")
    path = os.path.join(app.config['UPLOAD_FOLDER'], "asymmetry.jpeg")
    image = cv2.imread(path)
    print(np.asarray(image,dtype="float32"))
    #data = pickle.loads(open("face_enc", "rb").read())
    #test_image = image.resize((100,100),image)
    test_image = cv2.resize(image, (100,100))
    print(np.asarray(test_image, dtype ="float32"))
    test_image = img_to_array(test_image)
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image,axis=0)
    class_labels = ['Normal','Asymmetrical']
    predictions = classifier.predict(test_image)[0]
    label = class_labels[predictions.argmax()]
    print(label)

    if "Asymmetrical" in label:
        return render_template("buttons.html")
    elif "Normal" in label:
        return render_template("buttons.html")
    else:
        return render_template("homepage.html")
def classify_audio(): #slurred_speech
    classifier = load_model("/Users/erickacorral/Desktop/Revised-Code-Blue/Code-Blue/Revised-Code-Blue/flaskr/SlurredSpeechDetection.h5")
    path = os.path.join(app.config['UPLOAD_FOLDER'], "M03_Session2_0145.wav")
    audio, sample_rate = librosa.load(path) 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
   # mfccs_scaled_features_test = sc.transform(mfccs_scaled_features)
    mfccs_scaled_features_test=mfccs_scaled_features.reshape(mfccs_scaled_features.shape[0],16,8,1)
    #mfccs_scaled_features_test=mfccs_scaled_features_test.reshape(mfccs_scaled_features_test.shape[0],16,8,1) 
    class_labels = ['Slurred','Normal'] #not entirely sure this was the way it was encoded, may need to swap, but pretty sure
    predictions = classifier.predict(mfccs_scaled_features_test)[0]
    label = class_labels[predictions.argmax()]
    print(label)
    if "Normal" in label:
            return render_template("normal.html")
    elif "Slurred" in label:
            return render_template("buttons.html")
    else:
            return render_template("homepage.html")
  	
#routing

@app.route('/')
def index():
    return render_template('homepage.html')

"""
@app.route('/upload')
def upload_file(): # type: ignore
   return render_template('upload.html')
"""

@app.route('/uploader', methods = ['POST'])
def uploader_file():
    f = request.files['file']
    f.save(secure_filename(f.filename))
    if "wav" in f.filename:
                return classify_audio()
    else:
                return predict_asymmetry_upload()
   # if this works out, I want to replace this with return predict_asymmetry_upload 
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def predict_asymmetry():
    """ Face detection section """
    # Load stroke detection model
    #os.path.join(os.curdir, "StrokeDetectionModel.h5")
    classifier = load_model("/Users/erickacorral/Desktop/Revised-Code-Blue/Code-Blue/Revised-Code-Blue/flaskr/StrokeDetectionModel.h5")


    # Set the facial detection algorithm
    casc_path_face = "/Users/erickacorral/Desktop/Revised-Code-Blue/Code-Blue/Revised-Code-Blue/flaskr/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(casc_path_face)
    #data = pickle.loads(open("face_enc", "rb").read())

    # Preprocess the face
    ret, frame = camera.read()
    labels = []
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(img,1.3,5)
    
    # Classify the picture
    class_labels = ['Normal','Asymmetrical']
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_color = img[y:y+h,x:x+w]
        roi_color = cv2.resize(roi_color,(100,100),interpolation=cv2.INTER_AREA)

        if np.sum([roi_color])!=0:
            roi = roi_color.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),3)
            time.sleep(5)
            if "Asymmetrical" in label:
                return render_template("buttons.html")
            elif "Normal" in label:
                return render_template("normal.html")
        else:
            cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),3)

@app.route('/resources')
def resources():
   return render_template('resources.html')

"""@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)"""

def allowed_file(filename):
    return '.' in filename and \
    filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/signup', methods = ['POST', 'GET' ] )
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        return redirect(url_for('login'))
    return render_template('signup.html', form = form)

@app.route('/logout')
def logout():
    return render_template('logout.html')

#Log in and sign up

@app.route('/login', methods = ['POST', 'GET'])
def login():
    if current_user.is_authenticated:
        return redirect('/upload')
     
    if request.method == 'POST':
        email = request.form['email']
        user = UserModel.query.filter_by(email = email).first()
        if user is not None and user.check_password(request.form['password']):
            login_user(user)
            return redirect('/login')
     
    return render_template('login.html')

@app.route('/register', methods=['POST', 'GET'])
def register():
    if current_user.is_authenticated:
        return redirect('/login')
     
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        emergency_contact = request.form['emergency_contact']
             
        user = UserModel(email=email, username=username, emergency_contact=emergency_contact)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html')


@app.route('/buttons', methods = ['POST', 'GET'])
def buttons():
    sent = ' '
    called = ' '
    if request.method == 'POST':
        exec(open("/Users/erickacorral/Desktop/Revised-Code-Blue/Code-Blue/Revised-Code-Blue/flaskr/send_sms.py").read())
        sent = "Sent"
    if request.method == 'POST':
       exec(open("/Users/erickacorral/Desktop/Revised-Code-Blue/Code-Blue/Revised-Code-Blue/flaskr/make_call.py").read())
       called = 'Called'
    return render_template('buttons.html')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploader_file',
                                    filename=filename),code=307)
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug = True)



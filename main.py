from flask import Flask,render_template,request,Response
import numpy as np
import cv2
import pickle
filename="finalized_mod.sav"
loaded_model = pickle.load(open(filename, 'rb'))
app=Flask(__name__)
camera = cv2.VideoCapture(0)
def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  #
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/fault')
def fault():
    return render_template('fault.html')



@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/cam')
def cam():
    """Video streaming home page."""
    return render_template('cam.html')
@app.route('/control')
def control():
    return render_template('control.html')
@app.route('/fault',methods=['GET','POST'])
def form():
    if(request.method=='POST'):
        va=request.form.get('va')
        vb=request.form.get('vb')
        vc=request.form.get('vc')
        ia=request.form.get('ia')
        ib=request.form.get('ib')
        ic=request.form.get('ic')
        va=int(va)
        vb=int(vb)
        vc=int(vc)
        ia=int(ia)
        ib=int(ib)
        ic=int(ic)
        va=va/500e3
        vb=vb/500e3
        vc=vc/500e3
        ia=ia/200
        ib=ib/200
        ic=ic/200

        arr=[]
        arr.append(va)
        arr.append(vb)
        arr.append(vc)
        arr.append(ia)
        arr.append(ib)
        arr.append(ic)
        arr=np.array(arr)
        arr1=arr.reshape(1,6)
        y = loaded_model.predict(arr1)
        t = y[0]
        if(t==0):
            return render_template('res.html',res="no")
        if(t==1):
            return render_template('res.html',res="single line ground")
        if(t==2):
            return render_template('res.html',res="double line ground")
        if(t==3):
            return render_template('res.html',res="line to line")
        if(t==5):
            return render_template('res.html',res="symmetrical")


    return render_template('fault.html')
if __name__ == '__main__':
    app.run(debug=True)
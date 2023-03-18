import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

st.set_page_config(layout="wide")

st.title("Stock Vision")
colx, coly = st.columns(2)
model = None
def load_model():
  model=tf.keras.models.load_model('./StockVision.h5', compile= False)
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()
with colx:
    st.write(""" ## Predict stocks using computer vision    \n
     Please input an image which contains a graph in it like the example image. \n""")
with coly:
    with st.expander("See sample input image"):
        st.image(Image.open("./Udemy.png"), caption="Example of input image")

st.set_option('deprecation.showfileUploaderEncoding', False)


scaler = MinMaxScaler(feature_range=(1,8))
start_point = -1
time_step = 224

def dataset(dataset, time_step):
    X = []
    Y = []
    for i in range (len(dataset)-time_step -1):
        X.append(dataset[i:(i+time_step)])
        Y.append(dataset[i + time_step])
    return np.array(X), np.array(Y)

def transform(i, final = 0):
    global points, start_point
    img = Image.open(pred_path)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,50,50])
    upper_red = np.array([220,255,255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(img,img, mask= mask)
    BWimg = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    cntrs, _ = cv2.findContours(BWimg, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    maxEdge = 0
    index = 0
    for idx,cntr in enumerate(cntrs):
        if len(cntr) > maxEdge:
            maxEdge = len(cntr)
            index = idx
    d = {}
    for i in cntrs[index]:
        ele = i[0][0]
        if ele not in d:
            d[ele] = np.empty((0,1))
        d[ele] = np.append(d[ele], i[0][1])
    l = np.empty((0,2))
    for i in d:
        l =  np.append(l, np.array( [[ i,min(d[i]) ]]) , axis=0)
    if final > 0:
      start_point = max(l[:,0]) - time_step
    l = scaler.fit_transform(l)
    points = np.append(points, l, axis = 0)

pred_path = st.file_uploader("Please upload a stock graph image file", type=["jpg", "png"])

if pred_path is None:
    st.text("Please upload a graph of any stock")
else:
    col1, col2 = st.columns(2)
    new_img = Image.open(pred_path)
    with col1:
        st.image(new_img, use_column_width=True)
        new_img  = cv2.cvtColor(np.array(new_img), cv2.COLOR_BGR2RGB)

    try:
        points = np.empty((0,2))
        transform(pred_path, final = 1)
        
        corrected_points = points[points[:, 0].argsort()]
        X_pred, Y_pred = dataset(points, time_step=time_step)
                                    
        value = corrected_points[:,1]
        dummy = value.copy()
        df = pd.DataFrame(dummy, columns=['Price'])
        x_pred = np.array(df['Price'])
        X_pred, y_pred = dataset(x_pred, time_step)
        # out = (new_img.shape[1] - len(df.iloc[:,0]))//4 - 10
        out = new_img.shape[1]
        # print("out", out)
        ans = np.array(df.iloc[:time_step,0])
        y0 = scaler.inverse_transform(np.array([0,df['Price'].iloc[-1]]).reshape(1, -1))[0][1]
        color_pred = {'red': (255, 0, 0), 'green': (0, 255, 0)}
        color = (127, 127, 127)
        new_img = cv2.line(new_img, [int(start_point + time_step - 100), int(y0)], [1000, int(y0)], color = color, thickness = 1)
        pred = []
        with col2:
            with st.spinner('Image is being generated'):
                generated_img = st.empty()
                pred = []
                x = int(start_point) + time_step
                new_points = []
                jump = 1
                prev_point = [x-3, int(scaler.inverse_transform([[0,df['Price'].tail(1).item()]])[0,1])]
                p_point = prev_point.copy()
                new_img  = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
                while True:
                # while out:
                    y = model.predict(ans[-time_step:].reshape(1, -1), verbose = 0).item()
                    ans = np.append(ans, y)
                    pred.append(y)
                    Y = int(scaler.inverse_transform([[0,y]])[0,1])
                    x += 1
                    new_points.append([int(x+(jump*2)),Y])
                    new_img = cv2.line(new_img,prev_point,new_points[-1], (200, 87, 200),1)
                    prev_point = new_points[-1]
                    jump += 1
                    generated_img.image(new_img)
                    if int(x+jump*2) > out:
                        break
                last_out = int(scaler.inverse_transform([[0, pred[-1]]])[0][1])
                if last_out < y0:
                    color = color_pred['green']
                elif last_out > y0:
                    color = color_pred['red']

                new_points = []
                prev_point = p_point
                x = int(start_point) + time_step
                for i in pred:
                    i = int(scaler.inverse_transform([[0, i]])[0][1])
                    new_points.append([x,i])
                    x += 1
                jump = 1
                for x,y in new_points:
                    new_img = cv2.line(new_img, prev_point, [int(x+jump*2),y], color, 2)
                    prev_point = [x+int(jump*2),y]
                    jump += 1
                generated_img.image(new_img)

            with col1:
                suggestion = None
                if (y0 - last_out) >= 20:
                    suggestion = "This is a strong buy recommendation. Don't miss this opportunity!"
                    color = 'green'
                elif (y0 - last_out) > 0:
                    suggestion = "Profit margin is quite low."
                    color = 'green'
                else:
                    suggestion = "The particular stock is not going to do well for the next time period.<br/>Not recommended to buy now."
                    color = 'red'

                st.markdown(f"<h4 style='color: {color};'>{suggestion}</h4>", unsafe_allow_html=True)

    except:
        st.write('# Please input an image which contains a graph which can be forecasted')
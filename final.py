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

model = None
def load_model():
  model=tf.keras.models.load_model('./StockVision.h5', compile= False)
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write(""" ## Predict stocks using computer vision""")

st.set_option('deprecation.showfileUploaderEncoding', False)


scaler = MinMaxScaler(feature_range=(1,8))
start_point = -1
time_step = 224

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
    new_img = np.array(new_img)

    try:
        points = np.empty((0,2))
        transform(pred_path, final = 1)
        
        corrected_points = points[points[:, 0].argsort()]
                                        
        value = corrected_points[:,1]

        dummy = value.copy()
        df = pd.DataFrame(dummy, columns=['Price'])

        ans = np.array(df.iloc[-time_step:,0])

        pred = []
        with col2:
            with st.spinner('Image is being generated'):
                while len(pred) < time_step//1.5:
                    y = model.predict(ans[-time_step:].reshape(1, -1), verbose = 0).flatten()
                    ans = np.append(ans, y)
                    pred.append(y)

        f1 = np.empty((0,2))
        for idx, val in enumerate(pred):
            f1 = np.append(f1, [[idx, val[0]]], axis = 0)
        pred = scaler.inverse_transform(f1)
        pred = pred[:,1]
        pred = np.array( np.round(pred),dtype='int64')

        new_points = []
        x = int(start_point) + time_step
        for i in pred:
            new_points.append([x,i])
            x += 1

        y0 = int(scaler.inverse_transform(np.array([0,df['Price'].iloc[-1]]).reshape(1, -1))[0][1])
        color_pred = {'red': (0, 0, 255), 'green': (0, 255, 0)}
        color = (127, 127, 127)
        if pred[-1] < y0:
            color = color_pred['green']
        elif pred[-1] > y0:
            color = color_pred['red']

        prev_point = tuple(new_points[0])

        begin = np.array([start_point + time_step-2, y0], dtype = int)
        new_img = cv2.line(new_img, prev_point, begin, color, 2)
        jump = 1
        for x,y in new_points:
            new_img = cv2.line(new_img, prev_point, [int(x*jump),y], color, 2)
            prev_point = [int(x*jump),y]

        new_img  = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.plot([start_point + time_step-100, 1000], [y0, y0], linestyle = 'dashed')
        plt.imshow(new_img)
        with col2:
            st.pyplot(plt)
        with col1:
            suggestion = None
            if pred[-1] > 2*y0:
                suggestion = "Strong buy/ hold the stock"
            elif pred[-1] > y0:
                suggestion = "Buy"
            else:
                suggestion = "Sell"
            st.write("### Recommendation:{suggestion}")
    except:
        st.write('# Please input an image which contains a graph which can be forecasted')
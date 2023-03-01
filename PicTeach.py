import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPooling2D, MaxPooling2D, Flatten, Dense

if "model_a" not in st.session_state:
    st.session_state.model_a = None

if "labels" not in st.session_state:
    st.session_state.labels = None

uploaded_files = {}

# 画像をリサイズする関数
def resize_image(image, size=28):
    return image.resize((size, size))

def run_training():
    pass

def create_model(labels, images):
    # ラベルと画像の前処理
    size = 28
    label_list = sorted(list(set(labels)))
    st.session_state.labels = {name: i for i, name in enumerate(label_list)}
    x_train = np.array([np.array(resize_image(img, size)) for img in images])
    y_train = np.array([st.session_state.labels[label] for label in labels])

    # モデルの定義
    model = keras.Sequential(
        [
            Rescaling(1./255, input_shape=(size, size, 3)),
            Conv2D(16, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(input_shape=(size, size)),
            Dense(len(label_list), kernel_initializer='glorot_uniform', bias_initializer='zeros')
        ]
    )

    # モデルのコンパイル
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # モデルの訓練
    st.write("学習中・・・")
    progress_bar = st.progress(0)
    ep = 100
    model.fit(x_train, y_train, epochs=ep)
    progress_bar.progress(100)
    st.write("完了！")

    return model

def predict(model, image):
    # 画像の前処理
    x = np.expand_dims(np.array(image), axis=0)

    # 予測ラベルの算出
    y_pred = model.predict(x)[0]
    label_index = np.argmax(y_pred)
    return label_index

num_sections = st.sidebar.number_input("ラベルを追加", min_value=1, max_value=5, value=1, step=1)

st.title("PicTeach")

for i in range(num_sections):
    label_button = st.sidebar.button(f"ラベル {i+1}")
    # st.write(f"ラベル {i+1}")
    label_input = st.text_input(f"ラベル {i+1}", f"ラベル{i+1}")
    files = st.file_uploader("Choose a file", accept_multiple_files=True, type=["jpg", "jpeg"], key=i)
    if files:
        i = 0
        num = 5
        col= st.columns(num)
        if label_input not in uploaded_files:
            uploaded_files[label_input] = []
        for file in files:
            image = Image.open(file)
            resized_image = resize_image(image, 100)
            with col[i]:
                st.write(file.name)
                st.image(resized_image)
                uploaded_files[label_input].append(resized_image)
                i = (i + 1) % num
    st.markdown('<hr style="border-top: 3px solid #bbb;">', unsafe_allow_html=True)

training_button = st.sidebar.button("学習開始")

if training_button:
    labels = []
    images = []
    for label, imgs in uploaded_files.items():
        for img in imgs:
            labels.append(label)
            images.append(img)
    st.session_state.model_a = create_model(labels, images)

if st.session_state.model_a:
    predict_image = st.file_uploader("Choose a file", type=["jpg", "jpeg"])

    if predict_image:
        image = Image.open(predict_image)
        resized_image = resize_image(image)
        result = predict(st.session_state.model_a, resized_image)
        result_name = [k for k, v in st.session_state.labels.items() if v == result][0]
        st.header(result_name)
        st.image(image)
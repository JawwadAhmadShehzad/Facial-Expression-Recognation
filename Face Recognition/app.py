import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, History
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import pandas as pd
import pickle

app = Flask(__name__)

filename = 'fer2013.csv'
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
names = ['emotion', 'pixels', 'usage']
df = pd.read_csv(filename, names=names, na_filter=False)
im = df['pixels']

def getData(filename):
    Y = []
    X = []
    first = True
    for line in open(filename):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y

X, Y = getData(filename)
num_class = len(set(Y))

N, D = X.shape
X = X.reshape(N, 48, 48, 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)

def my_model():
    model = Sequential()
    input_shape = (48, 48, 1)
    model.add(Conv2D(64, (5, 5), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_class))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())

    return model


model_path = 'model_filter.h5'
model_checkpoint = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
history = History()

try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except OSError:
    print(f"No file or directory found at {model_path}. Training the model.")
    model = my_model()

    h = model.fit(x=X_train,
                  y=y_train,
                  batch_size=64,
                  epochs=20,
                  verbose=1,
                  validation_data=(X_test, y_test),
                  shuffle=True,
                  callbacks=[model_checkpoint, history]
                 )

    with open('training_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

emotions_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_image(image_path):
    img = image.load_img(image_path, color_mode="grayscale", target_size=(48, 48))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0
    return x

def emotion_analysis(emotions):
    return {emotion: float(emotion_value) for emotion, emotion_value in zip(emotions_list, emotions)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'})
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'})

        image_path = 'temp_image.jpg'
        file.save(image_path)
        processed_image = preprocess_image(image_path)
        predictions = model.predict(processed_image)
        predicted_emotion_index = np.argmax(predictions[0])
        predicted_emotion = emotions_list[predicted_emotion_index]
        return jsonify({'predicted_emotion': predicted_emotion})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction'})

if __name__ == '__main__':
    app.run(debug=True)
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install joblib')
get_ipython().system('pip install xgboost')
import joblib
import json
print("Done.")

from IPython.utils import io

import numpy as np
import pywt
import cv2
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sn

import os
import shutil

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

print("Done.")

img = cv2.imread('/Users/anupamsivaram/Desktop/Image Classification Code/Raw Test Images/brad pitt headshot - Google Search/brad-pitt-7.jpg')
img.shape

plt.imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray.shape

gray

plt.imshow(gray, cmap='gray')

face_cascade = cv2.CascadeClassifier('/Users/anupamsivaram/Desktop/Image Classification Code/opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/anupamsivaram/Desktop/Image Classification Code/opencv/haarcascades/haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
faces

(x, y, w, h) = faces[0]
x, y, w, h

face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
plt.imshow(face_img)

cv2.destroyAllWindows()
for (x,y,w,h) in faces:
    face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, y:x+w]
    roi_color = face_img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
plt.figure()
plt.imshow(face_img, cmap='gray')
plt.show()

get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(roi_color, cmap='gray')

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    if img is None:
            print("Error reading image:")
            return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        print(f"No faces detected in image")
        return None
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) == 2:
            return roi_color     

original_image = cv2.imread('/Users/anupamsivaram/Desktop/Image Classification Code/Raw Test Images/brad pitt headshot - Google Search/brad-pitt-7.jpg')
plt.imshow(original_image)

cropped_image = get_cropped_image_if_2_eyes('/Users/anupamsivaram/Desktop/Image Classification Code/Raw Test Images/brad pitt headshot - Google Search/brad-pitt-7.jpg')
plt.imshow(cropped_image)

org_image_obstructed = cv2.imread('/Users/anupamsivaram/Desktop/Image Classification Code/brad pitt obstructed.png')
plt.imshow(org_image_obstructed)

cropped_image_no_2_eyes = get_cropped_image_if_2_eyes('/Users/anupamsivaram/Desktop/Image Classification Code/brad pitt obstructed.png')
cropped_image_no_2_eyes

path_to_data = '/Users/anupamsivaram/Desktop/Image Classification Code/dataset'
path_to_cr_data = '/Users/anupamsivaram/Desktop/Image Classification Code/dataset/cropped/'

import os
img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)

img_dirs;

if os.path.exists(path_to_cr_data):
    shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)

with io.capture_output() as captured:

    cropped_image_dirs = []

    for img_dir in img_dirs:
        count = 1
        celebrity_name = img_dir.split('/')[-1]
        print(celebrity_name)

        celebrity_file_names_dict[celebrity_name] = []

        cropped_folder = path_to_cr_data + celebrity_name
        if not os.path.exists(cropped_folder):
            os.makedirs(cropped_folder)
            cropped_image_dirs.append(cropped_folder)
            print("Generating cropped images in folder: ", cropped_folder)


        for entry in os.scandir(img_dir):
            if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                roi_color = get_cropped_image_if_2_eyes(entry.path)
                if roi_color is not None:
                    cropped_file_name = celebrity_name + str(count) + ".png"
                    cropped_file_path = os.path.join(cropped_folder, cropped_file_name)
                    cv2.imwrite(cropped_file_path, roi_color)
                    count += 1
                    celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
    pass

print("Done.")

def w2d(img, mode='haar', level=1):
    imArray = img
    #These are Datatype conversions. We'll convert the images to grayscale.
    imArray = cv2.cvtColor ( imArray, cv2.COLOR_RGB2GRAY )
    #Convert this data to float. 
    imArray = np.float32(imArray)
    imArray /= 255;
    # Compute the coefficients. 
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    
    #Process the coefficients. 
    coeffs_H=list(coeffs)
    coeffs_H[0] *=0;
    
    #Reconstruction.
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H = np.uint8(imArray_H)
    
    return imArray_H

im_har = w2d(cropped_image,'db1',5)
plt.imshow(im_har, cmap='gray')

class_dict = {}
count = 0

for celebrity_name in celebrity_file_names_dict.keys():
    if "cropped" not in celebrity_name.lower():
        class_dict[celebrity_name] = count
        count += 1

class_dict

X, y = [], []
for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        if img is None:
            continue
        scaled_raw_img = cv2.resize(img, (32,32))
        img_har = w2d(img,'db1',5)
        scaled_img_har = cv2.resize(img_har, (32,32))
        combined_img = np.vstack((scaled_raw_img.reshape(32*32*3,1),scaled_img_har.reshape(32*32,1)))
        X.append(combined_img)
        y.append(class_dict[celebrity_name])

len(X)

len(X[0])

X[0]

X = np.array(X).reshape(len(X),4096).astype(float)
X.shape

X[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

len(X_test)

print(classification_report(y_test, pipe.predict(X_test)))

len(y_test)


# # Model Building and Tuning

model_params = {
    # SVM
    'svm': {
        'model': SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [1, 10, 100, 1000],
            'svc__kernel': ['rbf', 'linear']
        }
    },
    # Random Forest
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators': [1, 5, 10]
        }
    },
    # Logistic Regression
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'logisticregression__C': [1, 5, 10]
        }
    },
    # Naive Bayes
    'naive_bayes': {
        'model': GaussianNB(),
        'params': {}
    },
    # Decision Tree
    'decision_tree': {
        'model': tree.DecisionTreeClassifier(random_state=1),
        'params': {
            'decisiontreeclassifier__max_depth': [None, 5, 10]
        }
    },
    # K-Nearest Neighbors
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'kneighborsclassifier__n_neighbors': [3, 5, 7],
            'kneighborsclassifier__weights': ['uniform', 'distance']
        }
    },
    # XGBoost
    'xgboost': {
        'model': XGBClassifier(),
        'params': {
            'xgbclassifier__learning_rate': [0.01, 0.1, 0.2],
            'xgbclassifier__n_estimators': [50, 100, 200]
        }
    }
}

y_train = np.array(y_train)

print(y_train.shape)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import pandas as pd

model_params = {
    'svm': {
        'model': SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [1, 10, 100, 1000],
            'svc__kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators': [1, 5, 10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'logisticregression__C': [1, 5, 10]
        }
    },
    'naive_bayes': {
        'model': GaussianNB(),
        'params': {}
    },
    'decision_tree': {
        'model': tree.DecisionTreeClassifier(random_state=1),
        'params': {
            'decisiontreeclassifier__max_depth': [None, 5, 10]
        }
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'kneighborsclassifier__n_neighbors': [3, 5, 7],
            'kneighborsclassifier__weights': ['uniform', 'distance']
        }
    },
    'xgboost': {
        'model': XGBClassifier(),
        'params': {
            'xgbclassifier__learning_rate': [0.01, 0.1, 0.2],
            'xgbclassifier__n_estimators': [50, 100, 200]
        }
    }
}

scores = []
best_estimators = {}

for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)  # Subtracting 1 to make class labels start from 0
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
df

best_estimators['svm'].score(X_test, y_test)

best_estimators['random_forest'].score(X_test, y_test)

best_estimators['logistic_regression'].score(X_test,y_test)


best_clf = best_estimators['svm']


cm = confusion_matrix(y_test, best_clf.predict(X_test))
cm

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

class_dict

get_ipython().system('pip install joblib')
import joblib
# We'll save the model as a pickle in a file.
joblib.dump(best_clf, 'saved_model.pkl')

import json
with open("class_dictionary.json","w") as f:
    f.write(json.dumps(class_dict))
print("Done")


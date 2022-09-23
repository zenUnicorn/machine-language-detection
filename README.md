# Machine Language Detection
Using machine learning for language detection

## What is language detection?

The initial stage in any pipeline for text analysis or natural language processing is language identification. All ensuing language-specific models will yield wrong results if the language of a document is incorrectly determined. Similar to what happens when an English language analyzer is used on a French document, errors at this step of the analysis might accumulate and provide inaccurate conclusions. Each document's language and any elements that are written in another language need to be identified. The language used in papers varies widely depending on the nation and culture.

```python
import re
import warnings
warnings.simplefilter("ignore")

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
```
### Importing the dataset

```python
df = pd.read_csv("Language Detection.csv")
df.head()
```

```python
df.shape
```

```python
df["Language"].value_counts()
```
### Differentiating Independent from dependent features

```python
X = data["Text"]
y = data["Language"]
```
### Performing label encoding

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
```
### Text preparation

```python
text_list = []

for text in X:
        text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        text = text.lower()
        text_list.append(text)
```
### CountVectorizer

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(text_list).toarray()
X.shape # (10337, 39419)
```
### Train Test split

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
```
### Training and prediction of models

```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)
```


```python
y_prediction = model.predict(x_test)
```
### Model evaluation

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_prediction)
confusion_m = confusion_matrix(y_test, y_prediction)

print("The accuracy is :",accuracy)
```
### Visualization

```python
plt.figure(figsize=(15,10))
sns.heatmap(confusion_m, annot = True)
plt.show()
```
Let's try out the model prediction using text from several languages. We will write a function that will take in the text as an input and predict the language the text is written.

```python
def lang_predict(text):
     x = cv.transform([text]).toarray() 
     lang = model.predict(x)
     lang = le.inverse_transform(lang) 
     print("The langauge is in",lang[0]) 
```
### Testing...

```python
lang_predict("Today is going to be very busy because I have a lot of things to do.")
```
```python
lang_predict("سيكون اليوم مشغولاً للغاية لأن لدي الكثير من الأشياء لأفعلها.")
```
```python
lang_predict("आज का दिन बहुत व्यस्त रहने वाला है क्योंकि मेरे पास करने के लिए बहुत कुछ है।")
```
I will write an article about this project and add the link here.

Happy coding







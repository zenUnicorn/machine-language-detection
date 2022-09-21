# Machine Language Detection
Using machine learning for language detection


```python
import re
import warnings
warnings.simplefilter("ignore")

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
```

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

```python
X = data["Text"]
y = data["Language"]
```

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
```

```python
text_list = []

for text in X:
        text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        text = text.lower()
        text_list.append(text)
```

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(text_list).toarray()
X.shape # (10337, 39419)
```

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
```

```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)
```

```python
y_prediction = model.predict(x_test)
```

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_prediction)
confusion_m = confusion_matrix(y_test, y_prediction)

print("The accuracy is :",accuracy)
```

```python
plt.figure(figsize=(15,10))
sns.heatmap(confusion_m, annot = True)
plt.show()
```

```python
def lang_predict(text):
     x = cv.transform([text]).toarray() 
     lang = model.predict(x)
     lang = le.inverse_transform(lang) 
     print("The langauge is in",lang[0]) 
```
```python
lang_predict("Today is going to be very busy because I have a lot of things to do.")
```
```python
lang_predict("سيكون اليوم مشغولاً للغاية لأن لدي الكثير من الأشياء لأفعلها.")
```
```python
lang_predict("आज का दिन बहुत व्यस्त रहने वाला है क्योंकि मेरे पास करने के लिए बहुत कुछ है।")
```







# machine-language-detection
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

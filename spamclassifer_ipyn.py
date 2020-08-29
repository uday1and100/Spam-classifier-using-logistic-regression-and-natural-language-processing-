import pandas as pd
a = pd.read_csv("SMSSpamCollection",sep="\t",names=["labels","columns"])
import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
a["labels"].value_counts()

c=[]
for i in range(0,len(a["columns"])):
  d=[]
  x = re.sub("[^a-zA-Z]"," ",a["columns"][i])
  x = x.lower()
  b = nltk.word_tokenize(x)
  for j in range(0,len(b)):
    if b[j] not in set(stopwords.words("english")):
      d.append(lemmatizer.lemmatize(b[j]))
  b=" ".join(d)
  c.append(b)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
x = cv.fit_transform(c).toarray()

y = pd.get_dummies(a["labels"]).values
y = y[:,1]
print(y)
from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state=42)
x,y = smk.fit_sample(x,y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score
import matplotlib.pyplot as plt

lg = LogisticRegression()
lg.fit(x_train,y_train)
y_pred= lg.predict_proba(x_test)
thresholds = [0.5,0.6,0.65,0.7]
for i in thresholds:
  y_prede = y_pred[:,1]
  for j in range(0,len(y_prede)):
    if y_prede[j]>i:
      y_prede[j]=1
    else:
      y_prede[j]=0
  print("accuracy",accuracy_score(y_test,y_prede))
  print("precision",precision_score(y_test,y_prede))
  print("recall",recall_score(y_test,y_prede))
  print("\n")
   ##the best values are noted down in the lists pr and re
  pr = [0.92,0.95,0.96,0.98]
  re = [0.99,0.98,0.98,0.82]
  plt.plot(re,pr)
  plt.xlabel("precision")
  plt.ylabel("recall")
  plt.show()
i = 0.65 
#setting the threshold to 0.65
  y_prede = y_pred[:,1]
  for j in range(0,len(y_prede)):
    if y_prede[j]>i:
      y_prede[j]=1
    else:
      y_prede[j]=0
from sklearn.metrics import accuracy_score,precision_score,recall_score
print("accuracy",accuracy_score(y_test,y_prede))
print("precision",precision_score(y_test,y_prede))
print("recall",recall_score(y_test,y_prede))

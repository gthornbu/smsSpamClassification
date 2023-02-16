import sys
from packaging import version
import sklearn
import numpy as np
import pandas as pd

assert sys.version_info >= (3, 7)
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

np.random.seed(42)

df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])

df = df.drop_duplicates()

df['length']=df['message'].apply(len)

df.loc[:, 'label']=df.label.map({'ham':0, 'spam':1})


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#Count vectorizer converts text into a matrix of token counts
count = CountVectorizer()
text = count.fit_transform(df['message'])

x_train, x_test, y_train, y_test = train_test_split(text, df['label'], test_size=0.20, random_state=42)

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha=.01, fit_prior=True)
model.fit(x_train, y_train)

prediction = model.predict(x_test)

while True:  
    text = input("Please enter a message: ")
    new_text = [text]
    new_data = count.transform(new_text)
    y_pred = model.predict(new_data)
    print("Predicted label: ", y_pred)
    
    if y_pred[0] == 0:
        print("Predicted label for message is Ham")
    else:
        print("Predicted label for message is Spam")
    
    again = input("Do you want to classify another message? (y/n) ")
    if again.lower() != 'y':
        break

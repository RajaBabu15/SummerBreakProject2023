# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv('Data\\spam.csv',encoding = "windows-1252")

# %%
df.sample(10)

# %%
df.info()

# %%
df.shape

# %%
# TODO 
# 1. Data Cleaning 
# 2. EDA
# 3. Text Preprocessing
# 4. Model Building
# 5. Evaluation
# 6. Improvement
# 7. Website Building
# 8. Deployment of the Website



# %% [markdown]
# ## 1. Data Cleaning

# %%
df.info()

# %%
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

# %%
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)

# %%
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

# %%
df['target'] = encoder.fit_transform(df['target'])

# %%
df.isnull().sum()

# %%
df.duplicated().sum()

# %%
df.drop_duplicates(keep='first')

# %%
df.duplicated().sum()

# %%
df.shape

# %% [markdown]
# ## 2. EDA

# %%
df['target'].value_counts()

# %%
import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct='%0.2f')
plt.show()

# %%
#Data is imbalance

# %%
import nltk

# %%
df['num_characters'] = df['text'].apply(len)

# %%
df

# %%
#Fetch number of words
df['num_words'] = df['text'].apply(lambda x:len( nltk.word_tokenize(x)) )

# %%
df['num_sentences'] = df['text'].apply(lambda x:len( nltk.sent_tokenize(x)) )

# %%
df.head()

# %%
df[['num_characters','num_words','num_sentences']].describe()

# %%
#ham
df[df['target']==0][['num_characters','num_words','num_sentences']].describe()

# %%
#Spam Messages
df[df['target']==1][['num_characters','num_words','num_sentences']].describe()

# %%
import seaborn as sns

# %%
plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'],color='red')

# %%
plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_words'])
sns.histplot(df[df['target']==1]['num_words'],color='red')

# %%
plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_sentences'])
sns.histplot(df[df['target']==1]['num_sentences'],color='red')

# %%
sns.pairplot(df,hue='target')

# %%
df.corr()

# %%
sns.heatmap(df.corr(),annot=True)

# %% [markdown]
# ## 3. Data Preprocessing
# - Lower Case
# - Tokenization
# - Removing Specical Characters
# - Removing stop words and punctuatioon
# - Stemming

# %%
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

def transform_text(text):
    text = nltk.word_tokenize(text.lower())
    text = [word for word in text if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation]
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text]
    return " ".join(text)


# %%
stopwords.words('english')

# %%
string.punctuation

# %%
df['transformed_text'] = df['text'].apply(transform_text)

# %%
df.head()

# %%
from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')

# %%
spam_wc = wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep = " "))   #spam

# %%
plt.figure(figsize=(15,6))
plt.imshow(spam_wc)

# %%
ham_wc = wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep = " "))   #ham

# %%
plt.figure(figsize=(15,6))
plt.imshow(ham_wc)

# %%
spam_corpus = [word for msg in df[df['target']==1]['transformed_text'] for word in msg.split()]

# %%
len(spam_corpus)

# %%
from collections import Counter
Counter(spam_corpus).most_common(30)

# %%
spam_df = pd.DataFrame(Counter(spam_corpus).most_common(30), columns=['word', 'count'])
spam_df = spam_df.melt(id_vars='word', value_vars='count', var_name='variable', value_name='value')
sns.barplot(x='word', y='value', data=spam_df)
plt.xticks(rotation = 'vertical')

# %%
ham_corpus = [word for msg in df[df['target']==0]['transformed_text'] for word in msg.split()] #ham

# %%
len(ham_corpus)

# %%
Counter(ham_corpus).most_common(30)

# %%
ham_df = pd.DataFrame(Counter(ham_corpus).most_common(30), columns=['word', 'count'])
ham_df = spam_df.melt(id_vars='word', value_vars='count', var_name='variable', value_name='value')
sns.barplot(x='word', y='value', data=spam_df)
plt.xticks(rotation = 'vertical')

# %% [markdown]
# ## 4. Model Building

# %%
# NB has best for the textual data
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer()

# %%
X = cv.fit_transform(df['transformed_text']).toarray()

# %%
X.shape

# %%
y = df['target'].values

# %%
y

# %%
from sklearn.model_selection import train_test_split

# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# %%
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

# %%
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# %% [markdown]
# ## DATA IS IMBALANCE SO PRECISION_SCORE MATTER RATHER THAN ACCURARCY

# %%
gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score   (y_test,y_pred1))
print(confusion_matrix (y_test,y_pred1))
print(precision_score  (y_test,y_pred1))

# %%
mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score   (y_test,y_pred2))
print(confusion_matrix (y_test,y_pred2))
print(precision_score  (y_test,y_pred2))

# %%
bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score   (y_test,y_pred3))
print(confusion_matrix (y_test,y_pred3))
print(precision_score  (y_test,y_pred3))

# %% [markdown]
# ### Again Analysing the Data using the Tfidf Vectorizer Rather than the Previous CountVectorizer

# %%
X = tfidf.fit_transform(df['transformed_text']).toarray()

# %%
X.shape

# %%
y = df['target'].values

# %%
y

# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# %%
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# %%
gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score   (y_test,y_pred1))
print(confusion_matrix (y_test,y_pred1))
print(precision_score  (y_test,y_pred1))

# %%
mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score   (y_test,y_pred2))
print(confusion_matrix (y_test,y_pred2))
print(precision_score  (y_test,y_pred2))

# %%
bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score   (y_test,y_pred3))
print(confusion_matrix (y_test,y_pred3))
print(precision_score  (y_test,y_pred3))

# %% [markdown]
# # TFIFD WITH MULTINOMIAL NAIVE BAYES IS BEST 
# Though the accuracy is not high but it does not matter the most as it is attaining the high precison score
# and Also we can see that the best part of the model is that it is giving is False Negative as 0 means "all the spam(False) are for sure classified as the spam(Negative)"

# %% [markdown]
# # Let not stick to this model only. Checking the other classifier present in the scikit-learn library

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB,ComplementNB,CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier,RandomForestClassifier,StackingClassifier,VotingClassifier
from xgboost import XGBClassifier,XGBRFClassifier

# %%
lrc = LogisticRegression    (solver = 'liblinear',penalty='l1')

svc = SVC(kernel='sigmoid',gamma=1.0)

bnb = BernoulliNB   ()
gnb = GaussianNB    ()
mnb = MultinomialNB ()

dtc = DecisionTreeClassifier(max_depth=5)

knc = KNeighborsClassifier     ()

abc = AdaBoostClassifier            (n_estimators=50,random_state=2)
bac = BaggingClassifier             (n_estimators=50,random_state=2)
etc = ExtraTreesClassifier          (n_estimators=50,random_state=2)
gbc = GradientBoostingClassifier    (n_estimators=50,random_state=2)
rfc = RandomForestClassifier        (n_estimators=50,random_state=2)


xgb = XGBClassifier  (n_estimator = 50,random_state=2)
xfb = XGBRFClassifier(n_estimator = 50,random_state=2)

# %%
clfs = {
    'LINEAR REGRESSION': lrc,
    'SUPPORT VECTOR CLASSIFIER' : svc,
    'BERNOULLI NAIVE BAYES': bnb,
    'GAUSSIAN NAIVE BAYES': gnb,
    'MULTINOMIAL NAIVE BAYES': mnb,
    'DECISION TREE CLASSIFIER': dtc,
    'K NEAREST NEIGHBORS CLASSIFIER': knc,
    'ADABOOST CLASSIFIER': abc,
    'BAGGING CLASSIFIER': bac,
    'EXTRA TREES CLASSIFIER': etc,
    'GRADIENT BOOSTING CLASSIFIER': gbc,
    'RANDOM FOREST CLASSIFIER': rfc,
    'XGBOOST CLASSIFIER': xgb,
    'XGBOOST RANDOM FOREST CLASSIFIER': xfb
}


# %%
from sklearn.metrics import precision_score,accuracy_score

# %%
def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accura = accuracy_score(y_test,y_pred)
    precis = precision_score(y_test,y_pred)
    return accura,precis

# %%
accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    print("For : ",name)
    accuracy,precision = train_classifier(clf,X_train,y_train,X_test,y_test)
    print("accuracy_score : ",accuracy)
    print("precision_score : ",precision)
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)

# %%
performance_df = pd.DataFrame({'Algorithms':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)

# %%
performance_df

# %%
performance_df1 = pd.melt(performance_df,id_vars='Algorithms')

# %%
performance_df1

# %%
sns.catplot(x = 'Algorithms',y = 'value',hue = 'variable' ,data=performance_df1,kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation = 'vertical')
plt.show()

# %% [markdown]
# # 4. Model Building

# %% [markdown]
# ## 4.1 Trying with diffent number of features in the TF-idf Vectorizer (BEST FIT = 3000) 

# %%
# NB has best for the textual data
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000)

# %%
X = tfidf.fit_transform(df['transformed_text']).toarray()

# %%
X.shape

# %%
y = df['target'].values

# %%
y

# %%
lrc = LogisticRegression    (solver = 'liblinear',penalty='l1')

svc = SVC(kernel='sigmoid',gamma=1.0)

bnb = BernoulliNB   ()
gnb = GaussianNB    ()
mnb = MultinomialNB ()

dtc = DecisionTreeClassifier(max_depth=5)

knc = KNeighborsClassifier     ()

abc = AdaBoostClassifier            (n_estimators=50,random_state=2)
bac = BaggingClassifier             (n_estimators=50,random_state=2)
etc = ExtraTreesClassifier          (n_estimators=50,random_state=2)
gbc = GradientBoostingClassifier    (n_estimators=50,random_state=2)
rfc = RandomForestClassifier        (n_estimators=50,random_state=2)


xgb = XGBClassifier  (n_estimator = 50,random_state=2)
xfb = XGBRFClassifier(n_estimator = 50,random_state=2)

# %%
clfs = {
    'LINEAR REGRESSION': lrc,
    'SUPPORT VECTOR CLASSIFIER' : svc,
    'BERNOULLI NAIVE BAYES': bnb,
    'GAUSSIAN NAIVE BAYES': gnb,
    'MULTINOMIAL NAIVE BAYES': mnb,
    'DECISION TREE CLASSIFIER': dtc,
    'K NEAREST NEIGHBORS CLASSIFIER': knc,
    'ADABOOST CLASSIFIER': abc,
    'BAGGING CLASSIFIER': bac,
    'EXTRA TREES CLASSIFIER': etc,
    'GRADIENT BOOSTING CLASSIFIER': gbc,
    'RANDOM FOREST CLASSIFIER': rfc,
    'XGBOOST CLASSIFIER': xgb,
    'XGBOOST RANDOM FOREST CLASSIFIER': xfb
}


# %%
accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    print("For : ",name)
    accuracy,precision = train_classifier(clf,X_train,y_train,X_test,y_test)
    print("accuracy_score : ",accuracy)
    print("precision_score : ",precision)
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)

# %%
performance_df_features_3000 = pd.DataFrame({'Algorithms':clfs.keys(),'Accuracy_3000':accuracy_scores,'Precision_3000':precision_scores})
performance_df_features_3000

# %%
performance_df = performance_df.merge(performance_df_features_3000,on='Algorithms')

# %%
performance_df

# %% [markdown]
# ## 4.2 Trying the Min Max Scaler (Doesnot Work)
# I have used the MinMaxScaler as the Scaler for the Data rather than StandardScaler because the StandardScaler will result in negative values and the NB Classifier Doesnot work on the negative Value

# %%
# NB has best for the textual data
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

# %%
X = tfidf.fit_transform(df['transformed_text']).toarray()

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# %%
X.shape

# %%
y = df['target'].values

# %%
y

# %%
from sklearn.model_selection import train_test_split

# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# %%
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB,ComplementNB,CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier,RandomForestClassifier,StackingClassifier,VotingClassifier
from xgboost import XGBClassifier,XGBRFClassifier

# %%
lrc = LogisticRegression    (solver = 'liblinear',penalty='l1')

svc = SVC(kernel='sigmoid',gamma=1.0)

bnb = BernoulliNB   ()
gnb = GaussianNB    ()
mnb = MultinomialNB ()

dtc = DecisionTreeClassifier(max_depth=5)

knc = KNeighborsClassifier     ()

abc = AdaBoostClassifier            (n_estimators=50,random_state=2)
bac = BaggingClassifier             (n_estimators=50,random_state=2)
etc = ExtraTreesClassifier          (n_estimators=50,random_state=2)
gbc = GradientBoostingClassifier    (n_estimators=50,random_state=2)
rfc = RandomForestClassifier        (n_estimators=50,random_state=2)


xgb = XGBClassifier  (n_estimator = 50,random_state=2)
xfb = XGBRFClassifier(n_estimator = 50,random_state=2)

# %%
clfs = {
    'LINEAR REGRESSION': lrc,
    'SUPPORT VECTOR CLASSIFIER' : svc,
    'BERNOULLI NAIVE BAYES': bnb,
    'GAUSSIAN NAIVE BAYES': gnb,
    'MULTINOMIAL NAIVE BAYES': mnb,
    'DECISION TREE CLASSIFIER': dtc,
    'K NEAREST NEIGHBORS CLASSIFIER': knc,
    'ADABOOST CLASSIFIER': abc,
    'BAGGING CLASSIFIER': bac,
    'EXTRA TREES CLASSIFIER': etc,
    'GRADIENT BOOSTING CLASSIFIER': gbc,
    'RANDOM FOREST CLASSIFIER': rfc,
    'XGBOOST CLASSIFIER': xgb,
    'XGBOOST RANDOM FOREST CLASSIFIER': xfb
}


# %%
from sklearn.metrics import precision_score,accuracy_score

# %%
accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    print("For : ",name)
    accuracy,precision = train_classifier(clf,X_train,y_train,X_test,y_test)
    print("accuracy_score : ",accuracy)
    print("precision_score : ",precision)
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)

# %%
performance_df_MinMaxScaler = pd.DataFrame({'Algorithms':clfs.keys(),'Accuracy_MinMaxScaler':accuracy_scores,'Precision_MinMaxScaler':precision_scores})

# %%
performance_df_MinMaxScaler

# %%
performance_df = performance_df.merge(performance_df_MinMaxScaler,on='Algorithms')

# %%
performance_df

# %% [markdown]
# ## 4.3 Voting Classifier

# %%
from sklearn.ensemble import VotingClassifier


svc = SVC(kernel='sigmoid',gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50,random_state=2)

# %%
voc = VotingClassifier(estimators=[('svm',svc),('nb',mnb),('et',etc)],voting='soft')

# %%
voc.fit(X_train,y_train)

# %%
y_pred_vc = voc.predict(X_test)
print('Accuracy_score',accuracy_score(y_test,y_pred_vc))
print('Precision',precision_score(y_test,y_pred_vc))

# %% [markdown]
# ## 4.4 Stacking
# In the voting Classifer each classifier have the equal weight But we can change their weightage by applying the stacking

# %%
estimators=[('svm',svc),('nb',mnb),('et',etc)]
final_estimator = RandomForestClassifier()

# %%
from sklearn.ensemble import StackingClassifier

# %%
clf = StackingClassifier(estimators=estimators,final_estimator=final_estimator)

# %%
clf.fit(X_train,y_train)


# %%
y_pred = clf.predict(X_test)
print('accuracy : ',accuracy_score(y_test,y_pred))
print('precision : ',precision_score(y_test,y_pred))

# %%
import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(voc,open('model.pkl','wb'))

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%




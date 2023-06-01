import streamlit as st
from pickle4 import pickle
import nltk

tfidf = pickle.load(open('vectorizer1.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Spam Classifier')

input = st.text_input('Enter the Message')

if st.button("Predict"):

    # 1. Preprocess
    from nltk.corpus import stopwords
    import string
    from nltk.stem.porter import PorterStemmer

    def transform_text(text):
        text = nltk.word_tokenize(text.lower())
        text = [word for word in text if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation]
        ps = PorterStemmer()
        text = [ps.stem(word) for word in text]
        return " ".join(text)

    transformed_text = transform_text(input)

    # 2. Vectorizer using the vectorizer
    vector_input = tfidf.transform([transformed_text]).toarray()

    # 3. Predict using the Model
    result = model.predict(vector_input)[0]

    # 4. Display Input
    if result==1:
        st.header(":red[Spam]  	:no_entry_sign:")
    else:
        st.header(":green[Not Spam]	 :ok:")


    st.text("Sample Text:")
    st.text("Spam , Did u find out what time the bus is at coz i need to sort some stuff out.")
    st.text("Not Spam , Dude ive been seeing a lotta corvettes lately")
    st.text("Spam , Congratulations ur awarded either a yrs supply of CDs from Virgin Records or a Mystery Gift GUARANTEED Call 09061104283 Ts&Cs www.smsco.net 1.50pm approx 3mins")
    st.text("Not Spam , Same here, but I consider walls and bunkers and shit important just because I never play on peaceful but I guess your place is high enough that it don't matter")
    st.text("Spam , PRIVATE! Your 2003 Account Statement for 07808 XXXXXX shows 800 un-redeemed S. I. M. points. Call 08719899217 Identifier Code: 41685 Expires 07/11/04")
    st.text("Not Spam , Hello. We need some posh birds and chaps to user trial prods for champneys. Can i put you down? I need your address and dob asap. Ta r")
    st.text("Spam , What do U want for Xmas? How about 100 free text messages & a new video phone with half price line rental? Call free now on 0800 0721072 to find out more!")
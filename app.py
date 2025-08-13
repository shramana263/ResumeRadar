import streamlit as st
import pickle
import re
import nltk
# from sklearn.feature_extraction.text import TfidfVectorizer

# tfidf = TfidfVectorizer(stop_words='english')


nltk.download('punkt')
nltk.download('stopwords')

#loading models

clf=pickle.load(open('clf.pkl','rb'))
tfidf=  pickle.load(open('tfidf.pkl','rb'))

#web app

def cleanResume(resume_text):
    cleanTxt= re.sub('http\S+\s',' ',resume_text)
    cleanTxt= re.sub('RT|cc',' ',cleanTxt)
    cleanTxt= re.sub('#\S+\s',' ',cleanTxt)
    cleanTxt= re.sub('@\S+',' ',cleanTxt)
    cleanTxt= re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',cleanTxt)
    cleanTxt= re.sub(r'[^\x00-\x7f]',' ',cleanTxt)
    cleanTxt= re.sub('\s+',' ',cleanTxt)
    
    
    return cleanTxt

def main():
    st.title("Resume Screening App")
    uploaded_file= st.file_uploader('Upload Resume', type=['pdf','txt'])

    if uploaded_file is not None:
        try:
            resume_bytes=uploaded_file.read()
            resume_text=resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            #if utf decoding fails try latin-1 decoding
            resume_text=resume_bytes.decode('latin-1')
            
            
        cleaned_resume= cleanResume(resume_text)
        input_features= tfidf.transform([cleaned_resume])
        prediction_id=clf.predict(input_features)[0]
        st.write("Prediction ID:", prediction_id)
        

#python main
if __name__ == "__main__":
    main()
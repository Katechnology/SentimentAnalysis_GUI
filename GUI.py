import streamlit as st
from joblib import load
import numpy as np
from scipy.sparse import hstack
import regex
from underthesea import word_tokenize, pos_tag, sent_tokenize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from wordcloud import WordCloud
df = pd.read_csv('EDA_source.csv')

hotel_profiles = pd.read_csv('hotel_profiles.csv')
# Load the necessary components
vectorizer = load('vectorizer.joblib')
scaler = load('scaler.joblib')
model = load('model.joblib') 
##LOAD EMOJICON
file = open('emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()
def process_text(text, emoji_dict, teen_dict, wrong_lst):
    document = text.lower()
    document = document.replace("’",'')
    document = regex.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        # ...
        ###### DEL wrong words
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence+ sentence + '. '
    document = new_sentence
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    #...
    return document
# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

def process_special_word(text):
    # có thể có nhiều từ đặc biệt cần ráp lại với nhau
    new_text = ''
    text_lst = text.split()
    i= 0
    # không, chẳng, chả...
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()
import re
# Hàm để chuẩn hóa các từ có ký tự lặp
def normalize_repeated_characters(text):
    # Thay thế mọi ký tự lặp liên tiếp bằng một ký tự đó
    # Ví dụ: "lònggggg" thành "lòng", "thiệtttt" thành "thiệt"
    return re.sub(r'(.)\1+', r'\1', text)
def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        ###### POS tag
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        # lst_word_type = ['A','AB','V','VB','VY','R']
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document
def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

def preprocess(text, emoji_dict, teen_dict, english_dict, wrong_lst, stopwords):
    text = text.lower()
    text = process_text(text, emoji_dict, teen_dict, wrong_lst)
    text = covert_unicode(text)
    text = process_special_word(text)
    text = normalize_repeated_characters(text)
    text = process_postag_thesea(text)
    text = remove_stopword(text, stopwords)
    return text

def predict_sentiment(review, score):
    preprocessed_review = preprocess(review, emoji_dict, teen_dict, english_dict, wrong_lst, stopwords_lst)
    X_text = vectorizer.transform([preprocessed_review])
    score_array = np.array([score]).reshape(-1, 1)
    X_final = hstack([X_text, score_array])
    X_scaled = scaler.transform(X_final)
    prediction = model.predict(X_scaled)
    return prediction

def visualize_hotel_data(df, hotel_id):
    # Filter DataFrame for specific hotel
    specific_hotel_df = df[df['Hotel ID'] == hotel_id]

    # Set up the figure and axes
    fig, axs = plt.subplots(3, 2, figsize=(20, 15)) 

    fig.suptitle('Nationality Distribution and Group Name Distribution', fontsize=16, y=0.93)
    # Nationality Distribution
    nationality_counts = specific_hotel_df['Nationality'].value_counts()
    nationality_counts.plot(kind='bar', color='skyblue', ax=axs[0, 0])
    axs[0, 0].set_title(f'Nationality Distribution for Hotel ID: {hotel_id}')
    axs[0, 0].set_xlabel('Nationality')
    axs[0, 0].set_ylabel('Number of Reviews')
    axs[0, 0].tick_params(axis='x', rotation=45)

    # Group Name Distribution
    group_name_counts = specific_hotel_df['Group Name'].value_counts()
    group_name_counts.plot(kind='bar', color='lightgreen', ax=axs[0, 1])
    axs[0, 1].set_title(f'Group Name Distribution for Hotel ID: {hotel_id}')
    axs[0, 1].set_xlabel('Group Name')
    axs[0, 1].set_ylabel('Number of Reviews')
    axs[0, 1].tick_params(axis='x', rotation=45)

    fig.text(0.5, 0.63, 'Room Type Distribution and Score Distribution', ha='center', va='center', fontsize=16)
    # Room Type Distribution
    room_type_counts = specific_hotel_df['Room Type'].value_counts()
    room_type_counts.plot(kind='bar', color='salmon', ax=axs[1, 0])
    axs[1, 0].set_title(f'Room Type Distribution for Hotel ID: {hotel_id}')
    axs[1, 0].set_xlabel('Room Type')
    axs[1, 0].set_ylabel('Number of Reviews')
    axs[1, 0].tick_params(axis='x', rotation=45)

    
    # Score Distribution
    specific_hotel_df['Score'].plot(kind='hist', bins=20, ax=axs[1, 1])
    axs[1, 1].set_title(f'Score Distribution for Hotel ID: {hotel_id}')
    axs[1, 1].set_xlabel('Score')
    axs[1, 1].set_ylabel('Frequency')
    
    fig.text(0.5, 0.33, 'Score Level Distribution and Sentiment Distribution', ha='center', va='center', fontsize=16)
    # Score Level Distribution
    score_level_counts = specific_hotel_df['Score Level'].value_counts()
    score_level_counts.plot(kind='bar', color='purple', ax=axs[2, 0])
    axs[2, 0].set_title(f'Score Level Distribution for Hotel ID: {hotel_id}')
    axs[2, 0].set_xlabel('Score Level')
    axs[2, 0].set_ylabel('Number of Reviews')
    axs[2, 0].tick_params(axis='x', rotation=45)

    # Sentiment Distribution
    sentiment_counts = specific_hotel_df['Sentiment'].value_counts()
    sentiment_counts.plot(kind='bar', color='orange', ax=axs[2, 1])
    axs[2, 1].set_title(f'Sentiment Distribution for Hotel ID: {hotel_id}')
    axs[2, 1].set_xlabel('Sentiment')
    axs[2, 1].set_ylabel('Count of Reviews')
    axs[2, 1].tick_params(axis='x', rotation=0)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.3)

    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    return fig
def visual_insight_informations(df, hotel_id):
    # Filter DataFrame for specific hotel
    specific_hotel_df = df[df['Hotel ID'] == hotel_id]
    
    # Prepare data
    specific_hotel_df['Review Date'] = pd.to_datetime(specific_hotel_df['Review Year'].astype(str) + '-' + specific_hotel_df['Review Month'].astype(str))

    # Set up the figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    # WordCloud from Review Bodies
    text = ' '.join(review for review in specific_hotel_df['Body'] if isinstance(review, str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    axs[0, 0].imshow(wordcloud, interpolation='bilinear')
    axs[0, 0].axis('off')
    axs[0, 0].set_title('Word Cloud from Review Bodies')

    # Scatter plot of Score vs Score Level
    crosstab = pd.crosstab(specific_hotel_df['Nationality'], specific_hotel_df['Score Level'])
    crosstab.plot(kind='bar', stacked=True, ax=axs[0, 1])
    axs[0, 1].set_title('Score Level Distribution by Nationality')
    axs[0, 1].set_xlabel('Nationality')
    axs[0, 1].set_ylabel('Count')
    axs[0, 1].legend(title='Score Level')
    axs[0, 1].tick_params(axis='x', rotation=45)

    # Line plot of Scores over time
    time_scores = specific_hotel_df.groupby('Review Date')['Score'].mean()
    axs[1, 0].plot(time_scores.index, time_scores, marker='o', linestyle='-')
    axs[1, 0].set_title('Average Score Over Time')
    axs[1, 0].set_xlabel('Date')
    axs[1, 0].set_ylabel('Average Score')
    axs[1, 0].grid(True)

    sns.boxplot(x='Nationality', y='Score', data=specific_hotel_df, ax=axs[1, 1])
    axs[1, 1].set_title('Score Distribution by Nationality')
    axs[1, 1].set_xlabel('Nationality')
    axs[1, 1].set_ylabel('Score')
    axs[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig

def main():
    st.title("Hotel and Sentiment Analysis")

    # Create a menu bar in the sidebar using selectbox
    activity = st.sidebar.selectbox("Choose an option:", ["Information about Project","About Your Hotel","Deep dive about your Hotel", "Predict Sentiment"])

    if activity == "Deep dive about your Hotel":
        with st.form(key='hotel_data_form'):
            cols = st.columns(2)  # Create two columns for inputs and buttons
            with cols[0]:
                hotel_id = st.text_input("Enter Hotel ID", "1_1")
            with cols[1]:
                show_data_button = st.form_submit_button("Show Data")
                show_insights_button = st.form_submit_button("Show Insights")
        if show_data_button:
            display_hotel_data(df, hotel_id)  # Assuming display_hotel_data is a defined function
        if show_insights_button:
            display_insight_data(df, hotel_id)

    elif activity == "Predict Sentiment":
        with st.form(key='sentiment_form'):
            review_input = st.text_area("Enter your review:")
            score_input = st.slider("Rating", 0.0, 10.0, 5.0, 0.1)
            submit_button = st.form_submit_button("Predict")
        if submit_button:
            display_sentiment_prediction(review_input, score_input)
    elif activity == "Information about Project":
        display_project_info()

    elif activity == "About Your Hotel":
        hotel_id = st.text_input("Enter Hotel ID to get information:", "")
        if st.button('Get Hotel Information'):
            display_hotel_profile(hotel_id)
def display_hotel_profile(hotel_id):
    if hotel_id:
        try:
            hotel_data = hotel_profiles[hotel_profiles['Hotel ID'] == hotel_id].iloc[0]
            
            st.markdown(f"### {hotel_data['Hotel Name']}")
            st.text(f"Xếp hạng : {hotel_data['Hotel Rank']}")
            st.text(f"Địa chỉ: {hotel_data['Hotel Address']}")
            st.text(f"Tổng điểm: {hotel_data['Total Score']}")
            st.markdown("### Chi tiết đánh giá")
            ratings = {
                'Vị trí': hotel_data['Vị trí'],
                'Độ sạch sẽ': hotel_data['Độ sạch sẽ'],
                'Dịch vụ': hotel_data['Dịch vụ'],
                'Tiện nghi': hotel_data['Tiện nghi'],
                'Đáng giá tiền': hotel_data['Đáng giá tiền'],
                'Sự thoải mái và chất lượng phòng': hotel_data['Sự thoải mái và chất lượng phòng']
            }
            for key, value in ratings.items():
                if pd.notna(value):
                    st.text(f"{key}: {value}")
        except IndexError:
            st.error("No data found for the given Hotel ID.")
    else:
        st.warning("Please enter a valid Hotel ID.")

def display_project_info():
    st.write("## Information about the Project")
    st.info("""
    **Project Overview:**
    This application, titled 'Hotel and Sentiment Analysis', leverages cutting-edge technology to offer insights into hotel data and predict sentiment from reviews. It aims to assist stakeholders in the hospitality industry to better understand customer feedback and improve services accordingly.
    
    **Objectives:**
    - **Visualize Hotel Data**: Enable users to query and visualize extensive data about hotels, helping in making informed decisions based on trends and stats.
    - **Predict Sentiment of Reviews**: Utilize natural language processing (NLP) techniques to analyze and predict the sentiment of hotel reviews, categorizing them into positive, neutral, or negative sentiments.
    - **User Engagement**: Provide an interactive interface for users to explore data analytics without needing prior experience with data science tools.

    **Technologies Used:**
    - **Streamlit**: For creating a seamless and interactive web application.
    - **Pandas**: To manage and manipulate data efficiently.
    - **Matplotlib/Seaborn**: For generating insightful data visualizations.
    - **Scikit-learn/TensorFlow**: Employed for building and deploying machine learning models for sentiment analysis.

    **Future Enhancements:**
    - Addition of real-time data updates.
    - Expansion of the dataset to include more geographic locations and languages.
    - Implementation of more advanced NLP models to improve accuracy in sentiment prediction.

    **Contact Information:**
    For more details, feedback, or contributions, please reach out via [Email](mailto:akaphan2302@gmail.com).

    This application is continuously evolving, and we welcome your input to make it even better!
    """)

def display_sentiment_prediction(review, score):
    prediction = predict_sentiment(review, score)
    html_template = "<div style='text-align: center; color: {}; font-size: 40px; margin-top: 20px;'>{}</div>"
    color = "red" if prediction[0] == 0 else "orange" if prediction[0] == 1 else "green"
    formatted_html = html_template.format(color, ["Bad Review", "Neutral Review", "Good Review"][prediction[0]])
    st.markdown(formatted_html, unsafe_allow_html=True)

def display_hotel_data(df, hotel_id):
    st.markdown("### Basic Statistics about your hotel")
    try:
        fig = visualize_hotel_data(df, hotel_id)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error displaying data: {e}")

def display_insight_data(df, hotel_id):
    st.markdown("### Insight about your hotel")
    try:
        fig = visual_insight_informations(df, hotel_id)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error displaying data: {e}")

if __name__ == '__main__':
    main()

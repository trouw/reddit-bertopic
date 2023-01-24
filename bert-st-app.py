from bertopic import BERTopic 
import streamlit as st
import pandas as pd
import praw
import json

st.title("Discover Topics from Reddit")

container = st.container()
container.write("This tool scrapes the comments for the associated posts within a subreddit on reddit and performs topic modeling on the results to organize. Users can pick between controversial, hot, new, rising, and top posts, as well as the number of posts they would like to pull.")
container.write("Created by [Tyler Rouwhorst](https://www.linkedin.com/in/tyler-rouwhorst/)")
container.subheader("Directions:")
container.write("1. Type subreddit name into provided field")
container.write("2. Select type of posts to retrieve")
container.write('3. Input the number of posts to retrieve')  

num = st.selectbox('HTML Element', ('100','200','300','400','500','None'))

list = ['100','200','300','400','500']
if num == list: 
  num_post = int(num)
else: 
  num_post = None

sub_name = st.text_input('Input Subreddit Name:')

x = 0
if st.button('run'):
  x = 1
if x == 1: 

    user_agent = "Topic Modeling"
    reddit = praw.Reddit(
        client_id="VKdd3WG5nTIrhuiSV1WUNQ",
        client_secret="QtwkHm0rAa6F6LuIkAzmgo91AFQT2A",
        user_agent=user_agent,
        check_for_async=False
    )

    sub_body = []

    for submission in reddit.subreddit(sub_name).top(limit=num_post):
        for comment in submission.comments:
            sub_body.append(str(comment.body))

    topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2")

    topics, probs = topic_model.fit_transform(sub_body)

    df = pd.DataFrame(topic_model.get_document_info(sub_body))

    csv = df.to_csv().encode('utf-8')

    if csv != None:                                                                        
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='FAQ-Schema.csv')


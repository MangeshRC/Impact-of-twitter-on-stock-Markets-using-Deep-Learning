import pandas as pd
import numpy as np
import tensorflow as tf
import os
# for some models GPU capacity was not enough hence trained those models by disabling GPU with below line
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
import tensorflow_text as text
import pathlib
import tensorflow_addons as tfa
from keras.models import load_model


def function1(X):
    '''
        1.  This function takes input in pandas dataframe format with following columns and predicts
            the % chnage in closing price and sign of change.
            
            ['ticker_symbol', 'post_date', 'body', 'comment_num', 'retweet_num','like_num']
        
        2.  The input can be of one date,company or multiple.
    '''
    # Extracting features from input
    
    # function to detect whether URL is present or not.
    def Find_url(string):  
        return ('https' in string or 'http' in string)

    X['URL_flag'] = X.body.apply(lambda x:1 if Find_url(x) else 0)
    
    # fuunction to know whether hashtags are present in tweet text or not
    def Find_hashtag(string):
        temp = re.search(r"#(\w+)", string)     
        return temp

    # Extracting hashtag_flag feature
    X['hastags_flag'] = X.body.apply(lambda x:1 if Find_hashtag(x) else 0)
    
    # referred and modified below link to extract hashtags from tweets
    # https://www.geeksforgeeks.org/python-extract-hashtags-from-text/
    def Find_mention(string):
        temp = re.search(r"@(\w+)", string)     
        return temp

    # extracting mention_flag features
    X['mention_flag'] = X.body.apply(lambda x:1 if Find_mention(x) else 0)
    
    # referred below link to extract hashtags from tweets
    # https://www.geeksforgeeks.org/python-extract-hashtags-from-text/

    def get_hashtag(string):
        hashtags  = re.findall(r"#(\w+)", string)
        return hashtags

    # extracting and storing hashtags in dataframe
    X['hashtags'] = X.body.apply(lambda x:','.join(get_hashtag(x)))
    
    # referred stopwords from below link
    # https://gist.github.com/sebleier/554280
    stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]



    # referred this cleaning function from Donor Choose assignments
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r"#(\w+)", '', text)
        text = re.sub(r"@(\w+)", '', text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = text.replace('\\r', ' ')
        text = text.replace('\\n', ' ')
        text = text.replace('\\"', ' ')
        text = re.sub('[^A-Za-z0-9]+', ' ', text)
        text = text.replace('\\r', ' ')
        text = text.replace('\\n', ' ')
        text = text.replace('\\"', ' ')
        text = ' '.join(e for e in text.split() if e.lower() not in stopwords)
        return text


    # cleaning tweet text and storing in 'tweet_cleaned' column
    X['tweet_cleaned'] = X.body.apply(lambda x:preprocess(x))
    
    sid = SentimentIntensityAnalyzer()

    senti_score_train = [sid.polarity_scores(x_body) for x_body in X['tweet_cleaned']]


    X['neg'] = [senti['neg'] for senti in senti_score_train]
    X['neu'] = [senti['neu'] for senti in senti_score_train]
    X['pos'] = [senti['pos'] for senti in senti_score_train]
    X['compound'] = [senti['compound'] for senti in senti_score_train]
    
    [retweet_num_scalar,comment_num_scalar,like_num_scalar] = pickle.load(open("scalars.pkl","rb"))

    X['retweet_num'] = retweet_num_scalar.fit_transform(X['retweet_num'].values.reshape(-1,1))
    X['comment_num'] = comment_num_scalar.fit_transform(X['comment_num'].values.reshape(-1,1))
    X['like_num'] = like_num_scalar.fit_transform(X['like_num'].values.reshape(-1,1))
    
    X['hashtags'] = X.hashtags.apply(lambda x:0 if len(x)<1 else x)
    
    keep_indices = []
    for i in range(X.shape[0]):
        if len(X.iloc[0]['tweet_cleaned'])>1:
            keep_indices.append(i)
    keep_indices = np.array(keep_indices)
    X = X.iloc[keep_indices]
    

    dates = X['post_date'].unique()
    companies = X['ticker_symbol'].unique()  

    # empty list to store combined tweet text
    tweets = {}

    for date in dates:
        tweet_data = X[X.post_date == date]
        # empty string to store and accumulate tweet text for a day
        all_tweets = ''
        # loop to iterate through all tweets on that day
        for tweet in tweet_data['tweet_cleaned']: 
            all_tweets += tweet

        # storing all tweets to tweets dictionary
        tweets[date] = all_tweets

    
    # referred this preprocess function from Donor Choose assignments
    def preprocess(text):
        text = text.lower()
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = text.replace('\\r', ' ')
        text = text.replace('\\n', ' ')
        text = text.replace('\\"', ' ')
        text = re.sub('[^A-Za-z0-9]+', ' ', text)
        return text

    # Getting Glove vec dictionary
    with open('glove_vectors', 'rb') as f:
        glove_dict = pickle.load(f)

    # Function to get hashtag vectors
    def vec_hashtag(hashtags):
        vec = np.zeros(300)   # empty vector of dimension 300
        n_letters = 1         # counting letters in all hashtags
        if hashtags:
            hashtags = hashtags.split(',')         # getting all hashtags for current tweet
            for hashtag in hashtags:               # loop to iterate through all hashtags
                hashtag = preprocess(hashtag)      # preprocessing all hashtags
                hashtag = hashtag.replace(" ", "") # removing spaces in hashtags
                for letter in hashtag:           
                    vec += glove_dict[letter]
                    n_letters += 1
            vec /= n_letters                       # finding average of all letter vectors
        return np.array(vec)

    # referred array padding from below two links
    # https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    # https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros

    feature_mat = {}         # emtpy list to store all feature matrices
    req_dim = 500            # selected tweets for a day
    for date in dates: # loop to iterate through all dates
        vec_data = X[X.post_date == date]    # filtering data for current date

        # getting feature vector for current date with all other features
        temp = vec_data.drop(['body','ticker_symbol','hashtags','post_date','tweet_cleaned'],axis=1).values

        # empty list to store hashtag vector
        hash_vec = []

        # loop to iterate thorugh all tweet hashtags and getting their vector representation
        for tweet in vec_data['hashtags'].values:
            hash_vec.append(vec_hashtag(tweet))
        hash_vec = np.array(hash_vec)

        # combining hashtag vectors and other feature vectors for current date
        hash_vec = np.hstack((temp,hash_vec))

        try:       # padding matrix with 0's if number of tweet vectors are less than req_dim
            hash_vec = np.pad(hash_vec, [(0, req_dim-hash_vec.shape[0]), (0, 0)])
        except:    # else select first 500 tweet vectors
            hash_vec = hash_vec[:req_dim,:]

        # adding feature matrix to defined list
        # added one more dimension so that while training neural network, these arrrays will be treated as single channel images
        feature_mat[date] = np.expand_dims(hash_vec,axis=-1)
    
    structured_data = pd.DataFrame(columns=['post_date','tweet_text','company_name','feat_mat'])
    
    for company in companies:
        for date in dates:
            structured_data = structured_data.append({'post_date':date,
                                    'tweet_text':tweets[date],
                                    'company_name':company,
                                    'feat_mat':feature_mat[date]},ignore_index=True)
    
    x_text = structured_data['tweet_text'].values
    x_feat = list(structured_data['feat_mat'].values)
    x_feat = np.array(x_feat)
    x_company = structured_data['company_name'].values
    
    bert_tokenizer_params=dict(lower_case=True)
    reserved_tokens=[]

    bert_vocab_args = dict(
        # The target vocabulary size
        vocab_size = 8000,
        # Reserved tokens that must be included in the vocabulary
        reserved_tokens=reserved_tokens,
        # Arguments for `text.BertTokenizer`
        bert_tokenizer_params=bert_tokenizer_params,
        # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
        learn_params={},
    )

    # creating BertTokenizer object with vocab text file genrated as per reference link stated above
    en_tokenizer = text.BertTokenizer('en_vocab.txt', **bert_tokenizer_params)
    vocab_size_text = len(pathlib.Path('en_vocab.txt').read_text().splitlines())+ 1
    
    # Using BertTokenizer to tokenize train data
    encoded_text = en_tokenizer.tokenize(x_text)
    encoded_text = encoded_text.merge_dims(-2,-1)   # reducing dimension of ragged tensor
    encoded_text = encoded_text.to_list()           # converting to list to pad the sequences
    max_length = 5000                               # max length of padding
    x_text = pad_sequences(encoded_text, maxlen=max_length, padding='post')
    
    tokenizer = pickle.load(open("tokenizer.pkl","rb"))
    # getting tokenized train data
    train_comp = np.array(tokenizer.texts_to_sequences(x_company))
    
    loaded_model = load_model("trained_model.h5")
    prediction = (x_company,loaded_model.predict([x_text,x_feat,train_comp]))
    
    prediction[1][0] = [i[0] for i in prediction[1][0]]
    prediction[1][1] = [1 if i[0]>=0.5 else 0 for i in prediction[1][1]]
    
    return prediction
	
	

def function2(X,Y1,Y2):
    '''
        1.  This function takes input in pandas dataframe format with following columns and gives outputs
            as mean deviation in chnage prediction and ratio of correct signs predicted.
            
            ['ticker_symbol', 'post_date', 'body', 'comment_num', 'retweet_num','like_num']
        
        2.  The input can be of multiple dates and companies.
    '''
    
    def Find_url(string):  
        return ('https' in string or 'http' in string)

    X['URL_flag'] = X.body.apply(lambda x:1 if Find_url(x) else 0)
    
    # fuunction to know whether hashtags are present in tweet text or not
    def Find_hashtag(string):
        temp = re.search(r"#(\w+)", string)     
        return temp

    # Extracting hashtag_flag feature
    X['hastags_flag'] = X.body.apply(lambda x:1 if Find_hashtag(x) else 0)
    
    # referred and modified below link to extract hashtags from tweets
    # https://www.geeksforgeeks.org/python-extract-hashtags-from-text/
    def Find_mention(string):
        temp = re.search(r"@(\w+)", string)     
        return temp

    # extracting mention_flag features
    X['mention_flag'] = X.body.apply(lambda x:1 if Find_mention(x) else 0)
    
    # referred below link to extract hashtags from tweets
    # https://www.geeksforgeeks.org/python-extract-hashtags-from-text/

    def get_hashtag(string):
        hashtags  = re.findall(r"#(\w+)", string)
        return hashtags

    # extracting and storing hashtags in dataframe
    X['hashtags'] = X.body.apply(lambda x:','.join(get_hashtag(x)))
    
    stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]



    # referred this cleaning function from Donor Choose assignments
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r"#(\w+)", '', text)
        text = re.sub(r"@(\w+)", '', text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = text.replace('\\r', ' ')
        text = text.replace('\\n', ' ')
        text = text.replace('\\"', ' ')
        text = re.sub('[^A-Za-z0-9]+', ' ', text)
        text = text.replace('\\r', ' ')
        text = text.replace('\\n', ' ')
        text = text.replace('\\"', ' ')
            # https://gist.github.com/sebleier/554280
        text = ' '.join(e for e in text.split() if e.lower() not in stopwords)
        return text


    # cleaning tweet text and storing in 'tweet_cleaned' column
    X['tweet_cleaned'] = X.body.apply(lambda x:preprocess(x))
    
    sid = SentimentIntensityAnalyzer()

    senti_score_train = [sid.polarity_scores(x_body) for x_body in tqdm(X['tweet_cleaned'])]


    X['neg'] = [senti['neg'] for senti in senti_score_train]
    X['neu'] = [senti['neu'] for senti in senti_score_train]
    X['pos'] = [senti['pos'] for senti in senti_score_train]
    X['compound'] = [senti['compound'] for senti in senti_score_train]
    
    X = X[X.retweet_num > 1].copy()
    [retweet_num_scalar,comment_num_scalar,like_num_scalar] = pickle.load(open("scalars.pkl","rb"))

    X['retweet_num'] = retweet_num_scalar.transform(X['retweet_num'].values.reshape(-1,1))
    X['comment_num'] = comment_num_scalar.transform(X['comment_num'].values.reshape(-1,1))
    X['like_num'] = like_num_scalar.transform(X['like_num'].values.reshape(-1,1))
    
    X['hashtags'] = X.hashtags.apply(lambda x:0 if len(x)<1 else x)
    
    keep_indices = []
    for i in range(X.shape[0]):
        if len(X.iloc[0]['tweet_cleaned'])>1:
            keep_indices.append(i)
    keep_indices = np.array(keep_indices)
    X = X.iloc[keep_indices]
    

    dates = X['post_date'].unique()
    companies = X['ticker_symbol'].unique()  

    # empty list to store combined tweet text
    tweets = {}

    for date in tqdm(dates):
        tweet_data = X[X.post_date == date]
        # empty string to store and accumulate tweet text for a day
        all_tweets = ''
        # loop to iterate through all tweets on that day
        for tweet in tweet_data['tweet_cleaned']: 
            all_tweets += tweet

        # storing all tweets to tweets dictionary
        tweets[date] = all_tweets

    
    # referred this preprocess function from Donor Choose assignments
    def preprocess(text):
        text = text.lower()
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = text.replace('\\r', ' ')
        text = text.replace('\\n', ' ')
        text = text.replace('\\"', ' ')
        text = re.sub('[^A-Za-z0-9]+', ' ', text)
        return text

    # Getting Glove vec dictionary
    with open('glove_vectors', 'rb') as f:
        glove_dict = pickle.load(f)

    # Function to get hashtag vectors
    def vec_hashtag(hashtags):
        vec = np.zeros(300)   # empty vector of dimension 300
        n_letters = 1         # counting letters in all hashtags
        if hashtags:
            hashtags = hashtags.split(',')         # getting all hashtags for current tweet
            for hashtag in hashtags:               # loop to iterate through all hashtags
                hashtag = preprocess(hashtag)      # preprocessing all hashtags
                hashtag = hashtag.replace(" ", "") # removing spaces in hashtags
                for letter in hashtag:           
                    vec += glove_dict[letter]
                    n_letters += 1
            vec /= n_letters                       # finding average of all letter vectors
        return np.array(vec)

    # referred array padding from below two links
    # https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    # https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros

    feature_mat = {}         # emtpy list to store all feature matrices
    req_dim = 500            # selected tweets for a day
    for date in tqdm(dates): # loop to iterate through all dates
        vec_data = X[X.post_date == date]    # filtering data for current date

        # getting feature vector for current date with all other features
        temp = vec_data.drop(['body','ticker_symbol','hashtags','post_date','tweet_cleaned'],axis=1).values

        # empty list to store hashtag vector
        hash_vec = []

        # loop to iterate thorugh all tweet hashtags and getting their vector representation
        for tweet in vec_data['hashtags'].values:
            hash_vec.append(vec_hashtag(tweet))
        hash_vec = np.array(hash_vec)

        # combining hashtag vectors and other feature vectors for current date
        hash_vec = np.hstack((temp,hash_vec))

        try:       # padding matrix with 0's if number of tweet vectors are less than req_dim
            hash_vec = np.pad(hash_vec, [(0, req_dim-hash_vec.shape[0]), (0, 0)])
        except:    # else select first 500 tweet vectors
            hash_vec = hash_vec[:req_dim,:]

        # adding feature matrix to defined list
        # added one more dimension so that while training neural network, these arrrays will be treated as single channel images
        feature_mat[date] = np.expand_dims(hash_vec,axis=-1)
    
    structured_data = pd.DataFrame(columns=['post_date','tweet_text','company_name','feat_mat'])
    
    for company in companies:
        for date in tqdm(dates):
            structured_data = structured_data.append({'post_date':date,
                                    'tweet_text':tweets[date],
                                    'company_name':company,
                                    'feat_mat':feature_mat[date]},ignore_index=True)
    x_dates = structured_data['post_date'].values
    x_text = structured_data['tweet_text'].values
    x_feat = list(structured_data['feat_mat'].values)
    x_feat = np.array(x_feat)
    x_company = structured_data['company_name'].values
    
    
    bert_tokenizer_params=dict(lower_case=True)
    reserved_tokens=[]

    bert_vocab_args = dict(
        # The target vocabulary size
        vocab_size = 8000,
        # Reserved tokens that must be included in the vocabulary
        reserved_tokens=reserved_tokens,
        # Arguments for `text.BertTokenizer`
        bert_tokenizer_params=bert_tokenizer_params,
        # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
        learn_params={},
    )

    # creating BertTokenizer object with vocab text file genrated as per reference link stated above
    en_tokenizer = text.BertTokenizer('en_vocab.txt', **bert_tokenizer_params)
    vocab_size_text = len(pathlib.Path('en_vocab.txt').read_text().splitlines())+ 1
    
    # Using BertTokenizer to tokenize train data
    encoded_text = en_tokenizer.tokenize(x_text)
    encoded_text = encoded_text.merge_dims(-2,-1)   # reducing dimension of ragged tensor
    encoded_text = encoded_text.to_list()           # converting to list to pad the sequences
    max_length = 5000                               # max length of padding
    x_text = pad_sequences(encoded_text, maxlen=max_length, padding='post')
    
    tokenizer = pickle.load(open("tokenizer.pkl","rb"))
    # getting tokenized train data
    train_comp = np.array(tokenizer.texts_to_sequences(x_company))
    
    loaded_model = load_model("trained_model.h5")
    prediction = (x_company,loaded_model.predict([x_text,x_feat,train_comp]))
    
    deviation = []

    count = 0
    sign_pred = [1 if i[0]>=0.5 else 0 for i in prediction[1][1]]
    for i,date,company in zip(range(len(x_dates)),x_dates,x_company):
        if Y1[date][company] == -1 or Y2[date][company] == -1:
            continue
        deviation.append(abs(prediction[1][0][i]-Y1[date][company]))
        if sign_pred[i] == Y2[date][company]:
            count += 1   
            
    print("Mean Deviation for % change in closing prices is : ",np.mean(deviation))
    print("Number of correct signs predicted out of {} are : {}".format(len(sign_pred),count))
    return [np.mean(deviation),count/len(sign_pred)]
	


'''email_preprocessor.py
Preprocess Enron email dataset into features for use in supervised learning algorithms
Michelle Phan
CS 251/2 Data Analysis Visualization
Spring 2023
'''
import re
import os
import numpy as np
import string


def tokenize_words(text):
    '''Transforms an email into a list of words.

    Parameters:
    -----------
    text: str. Sentence of text.

    Returns:
    -----------
    Python list of str. Words in the sentence `text`.

    This method is pre-filled for you (shouldn't require modification).
    '''
    # Define words as lowercase text with at least one alphabetic letter
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


def count_words(email_path='data/enron'):
    '''Determine the count of each word in the entire dataset (across all emails)

    Parameters:
    -----------
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_emails: int. Total number of emails in the dataset.

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use the `tokenize_words` function above to chunk it into a list of words.
    - Update the counts of each word in the dictionary.

    Hints:
    - Check out Python functions in the os and os.path modules for walking the directory structure.
    '''
    word_freq = dict()
    num_emails = 0
    if os.path.isdir(email_path):
        dirs = os.listdir(email_path)
        for dir in dirs:
            if dir != ".DS_Store":
                # get all the email file path
                # dir path
                cur_dir = os.path.join(email_path, dir)
                emails = os.listdir(cur_dir)
                # read each email file as a string
                for email in emails:
                    with open(os.path.join(cur_dir, email), 'r') as file:
                        # keep track num of emails in the dataset
                        num_emails += 1
                        # read each line
                        content = file.read()
                        # Remove all punctuations from the content
                        words = tokenize_words(content)
                        for word in words:
                            # add the word frequency in the dicttionary word_freq
                            if word not in word_freq:
                                word_freq[word] = 1
                            else:
                                word_freq[word] += 1
    return word_freq, num_emails
    


def find_top_words(word_freq, num_features=200):
    '''Given the dictionary of the words that appear in the dataset and their respective counts,
    compile a list of the top `num_features` words and their respective counts.

    Parameters:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_features: int. Number of top words to select.

    Returns:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    counts: Python list. Counts of the `num_features` words in high-to-low count order.
    '''
    top_words = []
    counts = []
    # sort dict by value in descending order
    sorted_word_freq = sorted(word_freq.items(), key=lambda x:x[1], reverse=True)
    converted_dict = dict(sorted_word_freq)
    
    # get the list of all keys and items
    print(sorted_word_freq)
    words = list(converted_dict.keys())
    word_counts = list(converted_dict.values())
    
    # remove 
    
    # only returns the first num_features keys and items
    top_words = words[: num_features]
    counts = word_counts[: num_features]
    
    return top_words, counts


def make_feature_vectors(top_words, num_emails, email_path='data/enron'):
    '''Count the occurance of the top W (`num_features`) words in each individual email, turn into
    a feature vector of counts.

    Parameters:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    num_emails: int. Total number of emails in the dataset.
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    feats. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam = 1 /ham = 0)

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use `tokenize_words` to chunk it into a list of words.
    - Count the occurance of each word, ONLY THOSE THAT APPEAR IN `top_words`.

    HINTS:
    - Start with your code in `count_words` and modify as needed.
    '''
    num_features = len(top_words)
    
    # a list to keep track of word count for top_words in each email
    feats = []
    # keep track of the current email
    num_emails = 0
    # keep track of class label for each email (spam = 1 /ham = 0)
    y = []
    if os.path.isdir(email_path):
        dirs = os.listdir(email_path)
        for dir in dirs:
            if dir != ".DS_Store":
                # get all the email file path
                # dir path
                cur_dir = os.path.join(email_path, dir)
                emails = os.listdir(cur_dir)
                # read each email file as a string
                for email in emails:
                    # keep track of class index for each email (spam = 1. ham = 0)
                    if dir == "spam":
                        y.append(1)
                    else:
                        y.append(0)
                        
                    # a new dict() for each email to only count the top_words in that email
                    word_freq = dict()
                    with open(os.path.join(cur_dir, email), 'r') as file:
                        # read each line
                        content = file.read()
                        # Remove all punctuations from the content
                        content = content.translate(str.maketrans('', '', string.punctuation))
                        # Convert all words to lowercase and split them into a list
                        words = content.lower().split()
                        
                        # perform wordcount
                        for word in words:
                            # add the word frequency in the dicttionary word_freq
                            if word in top_words:
                                if word not in word_freq:
                                    word_freq[word] = 0
                                word_freq[word] += 1
        
                        feats.append(word_freq)
                        
                        # keep track num of emails in the dataset
                        num_emails += 1
                     
    # convert feast into an ndarray A matrix
    # counts of all unique top_words in all emails
    # row = 1 email
    # column = 1 unique word/feature
    feats_temp = np.zeros((num_emails, num_features))
    for e in range(num_emails):
        keys = list(feats[e].keys())
        values = list(feats[e].values())
        for f in range(num_features):
            try:
                index = keys.index(top_words[f])
                feats_temp[e, f] = values[index]
            except:
                feats_temp[e, f] = 0
    feats = feats_temp
    
    # convert list of class label into ndarray
    y = np.array(y)

    return feats, y
    
    


def make_train_test_sets(features, y, test_prop=0.2, shuffle=True):
    '''Divide up the dataset `features` into subsets ("splits") for training and testing. The size
    of each split is determined by `test_prop`.

    Parameters:
    -----------
    features. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)
    test_prop: float. Value between 0 and 1. What proportion of the dataset samples should we use
        for the test set? e.g. 0.2 means 20% of samples are used for the test set, the remaining
        80% are used in training.
    shuffle: boolean. Should we shuffle the data before splitting it into train/test sets?

    Returns:
    -----------
    x_train: ndarray. shape=(num_train_samps, num_features).
        Training dataset
    y_train: ndarray. shape=(num_train_samps,).
        Class values for the training set
    inds_train: ndarray. shape=(num_train_samps,).
        The index of each training set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    x_test: ndarray. shape=(num_test_samps, num_features).
        Test dataset
    y_test:ndarray. shape=(num_test_samps,).
        Class values for the test set
    inds_test: ndarray. shape=(num_test_samps,).
        The index of each test set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    '''
    inds = np.arange(y.size)
    # shuffle the data
    if shuffle:
        features = features.copy()
        y = y.copy()

        inds = np.arange(y.size)
        np.random.shuffle(inds)
        features = features[inds]
        y = y[inds]

    # Your code here:
    # use test_prop to decide the index to slice the data
    num_emails = features.shape[0]
    
    # number of samples in the train set
    index = int((1 - test_prop) * num_emails)
    
    # construct the train set
    x_train = features[:index]
    y_train = y[:index]
    inds_train = inds[:index]
    
    # construct the test set
    x_test = features[index:]
    y_test = y[index:]
    inds_test = inds[index:]
    
    return x_train, y_train, inds_train, x_test, y_test, inds_test
    
    


def retrieve_emails(inds, email_path='data/enron'):
    '''Obtain the text of emails at the indices `inds` in the dataset.

    Parameters:
    -----------
    inds: ndarray of nonnegative ints. shape=(num_inds,).
        The number of ints is user-selected and indices are counted from 0 to num_emails-1
        (counting does NOT reset when switching to emails of another class).
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    Python list of str. len = num_inds = len(inds).
        Strings of entire raw emails at the indices in `inds`
    '''
    # a list to keep track of word count for top_words in each email
    feats = []
    # keep track of the current email
    num_emails = 0
    # keep track of class label for each email (spam = 1 /ham = 0)
    y = []
    if os.path.isdir(email_path):
        dirs = os.listdir(email_path)
        for dir in dirs:
            if dir != ".DS_Store":
                # get all the email file path
                # dir path
                cur_dir = os.path.join(email_path, dir)
                emails = os.listdir(cur_dir)
                # read each email file as a string
                for email in emails:
                    # keep track num of emails in the dataset
                    num_emails += 1
                    if num_emails - 1 not in inds:
                        continue
                    
                    with open(os.path.join(cur_dir, email), 'r') as file:
                        # read each line
                        content = file.read()
                        # Remove all punctuations from the content
                        content = content.translate(str.maketrans('', '', string.punctuation)).lower()
                        
                        feats.append(content)
                        
                        
    return feats

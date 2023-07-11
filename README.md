# Spam-Email-Classification

An email spam filter to determine whether an email is a spam (spam) or not (ham) by implementing 2 supervised learning algorithms KNN & Naive Bayes. 

## ☘️ Description
I implement and compare the performance of two supervised learning algorithms: K Nearest Neighbors (KNN) and Naive Bayes to spam filter the <a href="https://www.kaggle.com/datasets/wanderfj/enron-spam">Enron spam email dataset</a>  (~34,000 emails). I also compare KNN performance with $L^2$ and $L^1$ distance metrics



## 👀 Results:
1. KNN Performance

        Accuracy for KNN:  90.4%
        Confusion Matrix KNN:
         [[221.  12.]
         [ 36. 231.]]

2. Naive Bayes Performance

        Accuracy for Naive Bayes:  89.4%
        Confusion Matrix Naive Bayes:
         [[193.  40.]
         [ 13. 254.]]
3. Compare KNN performance with $L^2$ and $L^1$ distance metrics

        Accuracy for KNN with $L^2$ distance:  91.97%

        Confusion Matrix KNN with $L^2$ distance:
        
            [[3189.  143.]
            [ 381. 2812.]]
        
        Accuracy for KNN with $L^1$ distance:  92.61%
        
        Confusion Matrix KNN with $L^1$ distance:
        
            [[3174.  158.]
            [ 324. 2869.]]

## 📝 Analysis:

For the KNN, the accuracy is 90% while Naive Bayes' accuracy is 89%

The confusion matrix of KNN (with k = 2) is better than the confusion matrix for Naive Bayes because there are more samples in the diagonal in the matrix of KNN than that of Naive Bayes, which mean that more emails are correctly categorized as spam or ham.

* Pro of KNN:
    
    - KNN is a non-parametric algorithm which means it doesn't make any assumptions about the distribution of the data while Naive Bayes makes the assumption that the features are independent, which may not always be true in real-world scenarios.
    
    - in my case, KNN has higher accuracy than Naive Bayes

* Con of KNN: 
    
    - slow and computationally expensive, especially for large datasets and high-dimensional data while Naive Bayes is computationally efficient and fast, especially for high-dimensional data.
    
    - requires the entire training dataset to be present in memory to perform predictions.
    
    - is sensitive to the choice of k-value.

For KNN, $L^2$ distance is a better choice when the data has many irrelevant features and $L^1$ distance is also less affected by outliers than $L^2$ distance.
L2 distance is a good choice when the features are continuous. 

For spam email classification, $L^1$ distance may be a better choice than $L^2$ distance because spam email data may have many irrelevant features or features with different scales, such as the presence or absence of certain words or characters, the length of the email, etc. In spam email classification, there may be some emails that are very different from the majority of emails in the dataset, such as emails containing a large number of icons or links. Thus, we can see that the accuracy for $L^1$ is better than $L^2$ distance

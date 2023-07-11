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

        Accuracy for Naive Bayes:  0.894
        Confusion Matrix Naive Bayes:
         [[193.  40.]
         [ 13. 254.]]
3. Compare KNN performance with $L^2$ and $L^1$ distance metrics

        Accuracy for KNN with $L^2$ distance:  0.9196934865900384

        Confusion Matrix KNN with $L^2$ distance:
        
            [[3189.  143.]
            [ 381. 2812.]]
        
        Accuracy for KNN with $L^1$ distance:  0.9261302681992337
        
        Confusion Matrix KNN with $L^1$ distance:
        
            [[3174.  158.]
            [ 324. 2869.]]

## 📝 Analysis:

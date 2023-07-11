'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
MICHELLE PHAN
CS 251/2: Data Analysis Visualization
Spring 2023
'''
import numpy as np


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''
    def __init__(self, num_classes):
        '''Naive Bayes constructor

        TODO:
        - Add instance variable for `num_classes`.
        - Add placeholder instance variables the class prior probabilities and class likelihoods (assigned to None).
        You may store the priors and likelihoods themselves or the logs of them. Be sure to use variable names that make
        clear your choice of which version you are maintaining.
        '''
        pass
        self.num_classes = num_classes
        # class_priors: ndarray. shape=(num_classes,).
        self.class_priors = []
        #   Probability that a training example belongs to each of the classes
        #   For spam filter: prob training example is spam or ham
        self.prob = None

        # class_likelihoods: ndarray. shape=(num_classes, num_features).
        #   Probability that each word appears within class c
        self.class_likelihoods = None

    def get_priors(self):
        '''Returns the class priors (or log of class priors if storing that)'''
        return self.class_priors

    def get_likelihoods(self):
        '''Returns the class likelihoods (or log of class likelihoods if storing that)'''
        return self.class_likelihoods

    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        TODO:
        - Compute the class priors and class likelihoods (i.e. your instance variables) that are needed for
        Bayes Rule. See equations in notebook.
        '''
        # Compute the class priors (ham = 0 / spam = 1)
        self.class_priors = []

        for i in range(self.num_classes):
            self.class_priors.append(len(y[y == i]) / y.shape[0])
            
        # store log of the priors
        self.class_priors = np.log(self.class_priors)
        
        # class likelihoods
        self.class_likelihoods = np.zeros((self.num_classes, data.shape[1]))
        
        # for spam class
        for i in range(self.num_classes):
            each_count = data[y == i].sum(0)
            total_count = each_count.sum() + data.shape[1]
        
            likelihoods = (each_count + 1) / total_count
            self.class_likelihoods[i] = likelihoods
            
        # store the log of class likelihoods
        self.class_likelihoods = np.log(self.class_likelihoods)
        

    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        TODO:
        - For the test samples, we want to compute the log of the posterior by evaluating
        the the log of the right-hand side of Bayes Rule without the denominator (see notebook for
        equation). This can be done without loops.
        - Predict the class of each test sample according to the class that produces the largest
        log(posterior) probability (hint: this can also be done without loops).

        NOTE: Remember that you are computing the LOG of the posterior (see notebook for equation).
        NOTE: The argmax function could be useful here.
        '''
        posterior = self.class_priors + data @ self.class_likelihoods.T
        return np.argmax(posterior, axis = 1)

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        # count how many  the output labels are correct (y == y_pred)
        correct = np.sum(y == y_pred)
        # divide that by total number of labels to get the probability
        acc = correct / y.shape[0]
        return acc
        
    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Recall: the rows represent the "actual" ground truth labels, the columns represent the
        predicted labels.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        for i in range(y.shape[0]):
            confusion_matrix[y[i], y_pred[i]] += 1
            
        return confusion_matrix

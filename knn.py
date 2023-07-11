'''knn.py
K-Nearest Neighbors algorithm for classification
Michelle Phan
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from palettable import cartocolors

class KNN:
    '''K-Nearest Neighbors supervised learning algorithm'''
    def __init__(self, num_classes):
        '''KNN constructor

        TODO:
        - Add instance variable for `num_classes`
        '''
        # exemplars: ndarray. shape=(num_train_samps, num_features).
        #   Memorized training examples
        self.exemplars = None
        # classes: ndarray. shape=(num_train_samps,).
        #   Classes of memorized training examples
        self.classes = None

    def train(self, data, y):
        '''Train the KNN classifier on the data `data`, where training samples have corresponding
        class labels in `y`.

        Parameters:
        -----------
        data: ndarray. shape=(num_train_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_train_samps,). Corresponding class of each data sample.

        TODO:
        - Set the `exemplars` and `classes` instance variables such that the classifier memorizes
        the training data.
        '''
        self.exemplars = data
        self.classes = y

    def predict(self, data, k):
        '''Use the trained KNN classifier to predict the class label of each test sample in `data`.
        Determine class by voting: find the closest `k` training exemplars (training samples) and
        the class is the majority vote of the classes of these training exemplars.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network.
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_test_samps,). Predicted class of each test data
        sample.

        TODO:
        - Compute the distance from each test sample to all the training exemplars.
        - Among the closest `k` training exemplars to each test sample, count up how many belong
        to which class.
        - The predicted class of the test sample is the majority vote.
        '''
        y_pred = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            # Compute the distance from each test sample to all the training exemplars.
            dist = np.linalg.norm(data[i] - self.exemplars, axis = 1)
            # Only keep k closest  exemplars to each test sample
            k_indices = np.argsort(dist)[:k]
            
            # keep track of the corresponding class to k closest exemplars
            k_nearest_labels = self.classes[k_indices]
            
            # The predicted class of the test sample is the majority vote
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            max_count_idx = np.argmax(counts)
            y_pred[i] = unique[max_count_idx]
        return y_pred
        
    def predict_L1(self, data, k):
        '''Use the trained KNN classifier to predict the class label of each test sample in `data`.
        Determine class by voting: find the closest `k` training exemplars (training samples) and
        the class is the majority vote of the classes of these training exemplars.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network.
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_test_samps,). Predicted class of each test data
        sample.

        TODO:
        - Compute the distance from each test sample to all the training exemplars.
        - Among the closest `k` training exemplars to each test sample, count up how many belong
        to which class.
        - The predicted class of the test sample is the majority vote.
        '''
        y_pred = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            # Compute the distance from each test sample to all the training exemplars.
            # Compute the L1 distances between the test data point and all training data points
            dist = np.sum(np.abs(self.exemplars - data[i]), axis=1)
            # Only keep k closest  exemplars to each test sample
            k_indices = np.argsort(dist)[:k]
            
            # keep track of the corresponding class to k closest exemplars
            k_nearest_labels = self.classes[k_indices]
            
            # The predicted class of the test sample is the majority vote
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            max_count_idx = np.argmax(counts)
            y_pred[i] = unique[max_count_idx]
        return y_pred
        

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

    def plot_predictions(self, k, n_sample_pts):
        '''Paints the data space in colors corresponding to which class the classifier would
         hypothetically assign to data samples appearing in each region.

        Parameters:
        -----------
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.
        n_sample_pts: int.
            How many points to divide up the input data space into along the x and y axes to plug
            into KNN at which we are determining the predicted class. Think of this as regularly
            spaced 2D "fake data" that we generate and plug into KNN and get predictions at.

        TODO:
        - Pick a discrete/qualitative color scheme. We suggest, like in the clustering project, to
        use a ColorBrewer color palette. List of possible ones here:
        https://github.com/CartoDB/CartoColor/wiki/CARTOColor-Scheme-Names
            - An example: cartocolors.qualitative.Safe_4.mpl_colors
            - The 4 stands for the number of colors in the palette. For simplicity, you can assume
            that we're hard coding this at 4 for 4 classes.
        - Each ColorBrewer palette is a Python list. Wrap this in a `ListedColormap` object so that
        matplotlib can parse it (already imported above).
        - Make an ndarray of length `n_sample_pts` of regularly spaced points between -40 and +40.
        - Call `np.meshgrid` on your sampling vector to get the x and y coordinates of your 2D
        "fake data" sample points in the square region from [-40, 40] to [40, 40].
            - Example: x, y = np.meshgrid(samp_vec, samp_vec)
        - Combine your `x` and `y` sample coordinates into a single ndarray and reshape it so that
        you can plug it in as your `data` in self.predict.
            - Shape of `x` should be (n_sample_pts, n_sample_pts). You want to make your input to
            self.predict of shape=(n_sample_pts*n_sample_pts, 2).
        - Reshape the predicted classes (`y_pred`) in a square grid format for plotting in 2D.
        shape=(n_sample_pts, n_sample_pts).
        - Use the `plt.pcolormesh` function to create your plot. Use the `cmap` optional parameter
        to specify your discrete ColorBrewer color palette.
        - Add a colorbar to your plot
        '''
        # Pick a discrete/qualitative color scheme
        colorScheme = cartocolors.qualitative.Safe_4.mpl_colors
        cmap = ListedColormap(colorScheme)
        
        # Create a linspace for regularly spaced points between -40 and +40
        samp_vec = np.linspace(-40, 40, n_sample_pts)
        
        # get the x and y coordinates of the 2D "fake data" sample points
        # in square region from [-40, 40] to [40, 40]
        x, y = np.meshgrid(samp_vec, samp_vec)
        
        # Combine your `x` and `y` sample coordinates into a single ndarray
        # reshape it to shape=(n_sample_pts*n_sample_pts, 2) as your `data` in self.predict.
        xf = x.flatten()
        yf = y.flatten()
        pairs = np.column_stack((xf, yf))
        pairs = np.reshape(pairs, (n_sample_pts*n_sample_pts, 2))
        
        # Reshape the predicted classes (`y_pred`) to shape=(n_sample_pts, n_sample_pts)
        y_pred = self.predict(pairs, k)
        y_pred = np.reshape(y_pred, (n_sample_pts, n_sample_pts))
        
        # Create the plot
        # print(np.unique(y_pred.flatten()))
        plt.pcolormesh(y_pred, cmap = cmap)
        # plt.pcolormesh(y_pred)
        
        # Add a colorbar to your plot
        plt.colorbar()

        plt.title('k = {} and n_sample_pts = {}'.format(k, n_sample_pts))
        plt.tight_layout()
        
    def confusion_matrix(self, y, y_pred, k):
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
        confusion_matrix = np.zeros((k, k))
        for i in range(y.shape[0]):
            confusion_matrix[y[i], y_pred[i]] += 1
            
        return confusion_matrix

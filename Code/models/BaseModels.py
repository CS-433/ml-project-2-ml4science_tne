from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class LogisticRegressionModel():
    def __init__(self, use_pca=False, expl_var=1., penalty='l2', dual=False, solver='liblinear'):
        """Create a Logistic Regression (LR) model.

        Args:
            use_pca (bool, optional): whether or not to use PCA on the dataset. Defaults to False.
            expl_var (float, optional): percentage of explained variance to keep. Defaults to 1..
            penalty (str, optional): type of penalty used. Defaults to 'l2'.
            dual (bool, optional): whether or not to use the dual formulation. Defaults to False.
            solver (str, optional): name of the solver to use. Defaults to 'liblinear'.
        """
        self.model = LogisticRegression(penalty=penalty, dual=dual, solver=solver)
        self.scaler = StandardScaler()
        self.use_pca = use_pca
        if self.use_pca:
            self.expl_var = expl_var
            self.pca = PCA()

    def fit(self, X_train, y_train):
        """Fit the LR model using the train set and train labels.

        Args:
            X_train (df.DataFrame): train dataset.
            y_train (list): train labels.
        """
        X_train = self.scaler.fit_transform(X_train)

        if self.use_pca:
            self.pca = PCA(n_components=self.expl_var)
            X_train = self.pca.fit_transform(X_train)

        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Use the trained LR model to predict label given datapoints.

        Args:
            X_test (pd.DataFrame): dataset on which to predict the labels.

        Returns:
            list: list of predicted labels.
        """
        X_test = self.scaler.transform(X_test)
        if self.use_pca:
            X_test = self.pca.transform(X_test)

        return self.model.predict(X_test)
    
class SVMModel():
    def __init__(self, use_pca=False, expl_var=1., parameters={'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear', 'rbf', 'sigmoid']}):
        """Create a Support Vector Machine (SVM) model.

        Args:
            use_pca (bool, optional): whether or not to use PCA on the dataset. Defaults to False.
            expl_var (float, optional): percentage of explained variance to keep. Defaults to 1..
            parameters (dict, optional): dictionary of values for hyperparametrization.
            Defaults to {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear', 'rbf', 'sigmoid']}.
        """
        self.parameters = parameters
        self.model = GridSearchCV(SVC(), self.parameters)
        self.scaler = StandardScaler()
        self.use_pca = use_pca
        if self.use_pca:
            self.expl_var = expl_var
            self.pca = PCA()

    def fit(self, X_train, y_train):
        """Fit the SVM model using the train set and train labels.

        Args:
            X_train (df.DataFrame): train dataset.
            y_train (list): train labels.
        """
        X_train = self.scaler.fit_transform(X_train)

        if self.use_pca:
            self.pca = PCA(n_components=self.expl_var)
            X_train = self.pca.fit_transform(X_train)

        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Use the trained SVM model to predict label given datapoints.

        Args:
            X_test (pd.DataFrame): dataset on which to predict the labels.

        Returns:
            list: list of predicted labels.
        """
        X_test = self.scaler.transform(X_test)
        if self.use_pca:
            X_test = self.pca.transform(X_test)

        return self.model.predict(X_test)
    
class RandomForestModel():
    def __init__(self, use_pca=False, expl_var=1., parameters={'n_estimators': [10, 50, 90, 130], 'max_depth': [10, 25, 50]}):
        """Create a Random Forest (RF) model.

        Args:
            use_pca (bool, optional): whether or not to use PCA on the dataset. Defaults to False.
            expl_var (float, optional): percentage of explained variance to keep. Defaults to 1..
            parameters (dict, optional): dictionary of values for hyperparametrization.
            Defaults to {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear', 'rbf', 'sigmoid']}.
        """
        self.model = GridSearchCV(RandomForestClassifier(), parameters)
        self.scaler = StandardScaler()
        self.use_pca = use_pca
        if self.use_pca:
            self.expl_var = expl_var
            self.pca = PCA()

    def fit(self, X_train, y_train):
        """Fit the RF model using the train set and train labels.

        Args:
            X_train (df.DataFrame): train dataset.
            y_train (list): train labels.
        """
        X_train = self.scaler.fit_transform(X_train)

        if self.use_pca:
            self.pca = PCA(n_components=self.expl_var)
            X_train = self.pca.fit_transform(X_train)

        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Use the trained RF model to predict label given datapoints.

        Args:
            X_test (pd.DataFrame): dataset on which to predict the labels.

        Returns:
            list: list of predicted labels.
        """
        X_test = self.scaler.transform(X_test)
        if self.use_pca:
            X_test = self.pca.transform(X_test)

        return self.model.predict(X_test)

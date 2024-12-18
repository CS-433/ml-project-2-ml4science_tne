from utils import *
from dataset import *
from constants import *
from models.BaseModels import *
from models.DeepModels import *
from models.DeepUtils import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dataset import Participant
from torch.utils.data import DataLoader

# Reproducibility
seed_num = RANDOM_STATE # This seed will be used for all random number generators
torch.use_deterministic_algorithms(True) # PyTorch will use deterministic algorithms fro operations with stochastic behavior like dropout
random.seed(seed_num) # Python's random will use seed_num
np.random.seed(seed_num) # NumPy's random number generator will use seed_num
torch.manual_seed(seed_num) # PyTorch's random number will use seed_num

TEST_SIZE = 0.3
PCA_EXPL_VAR = 0.95

# MLP
LR = 0.1
EPOCHS = 10
WEIGHT_DECAY = 1e-3

def run_models(accuracies, features):
    """Run all models on the same features and stores accuracies in a dictionary.

    Args:
        accuracies (dict): prepared dictionary for the accuracies.
        features (pd.DataFrame): DataFrame containing the features.
    """
    
    X = features.drop('label', axis=1)
    y = features['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)  
    
    logreg = LogisticRegressionModel()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    accuracies['LR'].append(accuracy_score(y_test, y_pred))
    
    logreg = LogisticRegressionModel(use_pca=True, expl_var=PCA_EXPL_VAR)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    accuracies['LR PCA'].append(accuracy_score(y_test, y_pred))
    
    svm = SVMModel()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracies['SVM'].append(accuracy_score(y_test, y_pred))
    
    svm = SVMModel(use_pca=True, expl_var=PCA_EXPL_VAR)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracies['SVM PCA'].append(accuracy_score(y_test, y_pred))
    
    svm = RandomForestModel()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracies['RF'].append(accuracy_score(y_test, y_pred))
    
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
def plot_accuracy(accuracies, title, task, font_loc='best'):
    """Utility function to plot the accuracy of different models and participants.

    Args:
        accuracies (dict): dictionary of accuracies.
        title (str): title of the plot.
        task (str): indicates the classification task.
        font_loc (str, optional): location of the legend. Defaults to 'best'.
    """
    
    plt.rcParams["font.family"] = "Times New Roman"
    SMALL_SIZE = 20
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 28

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE, titleweight = 'bold', titlepad = 20)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE, labelweight = 'bold', labelpad = 15)   # fontsize of the x and y labels    
    plt.rc('xtick', labelsize=SMALL_SIZE, direction = 'out')    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE, direction = 'out')    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('axes.spines', top=False, right=False)  # Removing the top and right spines   

    models = accuracies.keys()

    dataset_values = np.array([accuracies[model] for model in models])

    x = np.arange(len(models))
    width = .8

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - 3*width/8, dataset_values[:,0], width/4, label='s6')
    bars2 = ax.bar(x - width/8, dataset_values[:,1], width/4, label='s7')
    bars3 = ax.bar(x + width/8, dataset_values[:,2], width/4, label='s11')
    bars4 = ax.bar(x + 3*width/8, dataset_values[:,3], width/4, label='s12')

    ax.set_xticks(x)
    ax.set_xticklabels(models)

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Models')
    ax.set_title(title)

    ax.set_ylim(0,1)

    ax.legend(loc=font_loc,ncols=2, fontsize='x-small')

    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x(), yval, round(yval, 2), va='bottom', fontsize=10)
        
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x(), yval, round(yval, 2), va='bottom', fontsize=10)
        
    for bar in bars3:
        yval = bar.get_height()
        ax.text(bar.get_x(), yval, round(yval, 2), va='bottom', fontsize=10)
        
    for bar in bars4:  
        yval = bar.get_height()
        ax.text(bar.get_x(), yval, round(yval, 2), va='bottom', fontsize=10)
        
    plt.axhline(y=0.5, color='k', linestyle='--', label='Chance level')
        
    plt.tight_layout()
    saved_dir = os.path.join(os.getcwd(), 'figures')
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    plt.savefig(f'figures/accuracies_across_part_{task}.png')



if __name__ == '__main__':
    
    # load.py needs to be run before this script
    for part_name in PARTICIPANTS:
        if not os.path.exists(f'saved/{part_name}.pkl'): raise FileNotFoundError(f'Participant {part_name} not found - Run load_data.py first')
    
    accuracies_ExObs = {'LR': [], 'LR PCA': [], 'SVM': [], 'SVM PCA': [], 'RF': []}
    accuracies_ex = {'LR': [], 'LR PCA': [], 'SVM': [], 'SVM PCA': [], 'RF': []}
    accuracies_obs = {'LR': [], 'LR PCA': [], 'SVM': [], 'SVM PCA': [], 'RF': []}

    for part_name in PARTICIPANTS:
        print(f'Processing participant {part_name}...')
        ex_features = pd.read_hdf(f'saved/ex_features_{part_name}_mvt.h5', 'df')
        obs_features = pd.read_hdf(f'saved/obs_features_{part_name}_mvt.h5', 'df')
        ExObs_features = pd.read_hdf(f'saved/features_{part_name}_ExObs.h5', 'df')
        
        print(f'Running models for participant {part_name}...')
        run_models(accuracies_ExObs, ExObs_features)
        run_models(accuracies_ex, ex_features)
        run_models(accuracies_obs, obs_features)
        
    print('Saving results...')
    saved_dir = os.path.join(os.getcwd(), 'saved')
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    with open('saved/accuracies_across_part_ex.pkl', 'wb') as f:
        pickle.dump(accuracies_ex, f, pickle.HIGHEST_PROTOCOL)
    with open('saved/accuracies_across_part_obs.pkl', 'wb') as f:
        pickle.dump(accuracies_obs, f, pickle.HIGHEST_PROTOCOL)
    with open('saved/accuracies_across_part_ExObs.pkl', 'wb') as f:
        pickle.dump(accuracies_ExObs, f, pickle.HIGHEST_PROTOCOL)
        
    plot_accuracy(accuracies_ex, 'Accuracies by model (execution)', 'ex')
    plot_accuracy(accuracies_obs, 'Accuracies by model (observation)', 'obs')
    plot_accuracy(accuracies_ExObs, 'Accuracies by model (action recognition)', 'ExObs', font_loc='lower left')
    print('Done! The plots are saved in the saved directory.')
    
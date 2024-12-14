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

random.seed(RANDOM_STATE)
TEST_SIZE = 0.3
PCA_EXPL_VAR = 0.95

# MLP
LR = 0.1
EPOCHS = 10
WEIGHT_DECAY = 1e-3

def run_models(accuracies, features):
    
    X = features.drop('label', axis=1)
    y = features['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)  
    
    logreg = LogisticRegressionModel()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    accuracies['lr'].append(accuracy_score(y_test, y_pred))
    
    logreg = LogisticRegressionModel(use_pca=True, expl_var=PCA_EXPL_VAR)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    accuracies['lr PCA'].append(accuracy_score(y_test, y_pred))
    
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
    accuracies['rf'].append(accuracy_score(y_test, y_pred))
    
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mlp = MLP(X_train.shape[1], 2, layers=(16, 16))
    trainset = DfDataset(X_train, y_train)
    valset = DfDataset(X_val, y_val)
    train_loader = DataLoader(trainset, batch_size=4, shuffle=True)
    val_loader = DataLoader(valset, batch_size=4, shuffle=False)

    trainer = Trainer(mlp, LR, EPOCHS, WEIGHT_DECAY, save_path='saved/mlp.pth', device=device)
    trainer.train(train_loader, val_loader)
    
    testset = DfDataset(X_test, y_test)
    acc = 0
    for input, label in testset:
        pred = trainer.model(input)
        if torch.argmax(pred) == label:
            acc += 1

    acc /= len(testset)
    accuracies['MLP'].append(acc)
    
def plot_accuracy(accuracies, title, task):
    
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

    dataset_values = [accuracies[model] for model in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title)

    bars1 = ax.bar(x - width/4, dataset_values[:,0], width, label='s6')
    bars2 = ax.bar(x - width/2, dataset_values[:,1], width, label='s7')
    bars3 = ax.bar(x + width/2, dataset_values[:,2], width, label='s11')
    bars4 = ax.bar(x + width/4, dataset_values[:,3], width, label='s12')

    ax.set_xticks(x)
    ax.set_xticklabels(models)

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Models')
    ax.set_title('Accuracies by model (execution)')

    ax.set_ylim(0,1)

    ax.legend()

    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/16, yval, round(yval, 2), va='bottom', fontsize=15)
        
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/16, yval, round(yval, 2), va='bottom', fontsize=15)
        
    for bar in bars3:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/16, yval, round(yval, 2), va='bottom', fontsize=15)
        
    for bar in bars4:  
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/16, yval, round(yval, 2), va='bottom', fontsize=15)
        
    plt.tight_layout()
    saved_dir = os.path.join(os.getcwd(), 'figures')
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    plt.savefig(f'figures/accuracies_across_part_{task}.png')

# load.py needs to be run before this script
if __name__ == '__main__':
    
    accuracies_ExObs = {'lr': [], 'lr PCA': [], 'SVM': [], 'SVM PCA': [], 'rf': [], 'MLP': []}
    accuracies_ex = {'lr': [], 'lr PCA': [], 'SVM': [], 'SVM PCA': [], 'rf': [], 'MLP': []}
    accuracies_obs = {'lr': [], 'lr PCA': [], 'SVM': [], 'SVM PCA': [], 'rf': [], 'MLP': []}

    for part_name in PARTICIPANTS:
        participant = Participant.load_from_pickle(f'saved/{part_name}.pkl')
        ex_features = pd.read_hdf(f'saved/ex_features_{part_name}_mvt.h5', 'df')
        obs_features = pd.read_hdf(f'saved/obs_features_{part_name}_mvt.h5', 'df')
        ExObs_features = pd.read_hdf(f'saved/features_{part_name}_ExObs.h5', 'df')
        
        run_models(accuracies_ExObs, ExObs_features)
        run_models(accuracies_ex, ex_features)
        run_models(accuracies_obs, obs_features)
        
    plot_accuracy(accuracies_ex, 'Accuracies by model (execution)', 'ex')
    plot_accuracy(accuracies_obs, 'Accuracies by model (observation)', 'obs')
    plot_accuracy(accuracies_ExObs, 'Accuracies by model (action recognition)', 'ExObs')
    
    saved_dir = os.path.join(os.getcwd(), 'saved')
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    with open('saved/accuracies_across_part_ex.pkl', 'wb') as f:
        pickle.dump(accuracies_ex, f, pickle.HIGHEST_PROTOCOL)
    with open('saved/accuracies_across_part_obs.pkl', 'wb') as f:
        pickle.dump(accuracies_obs, f, pickle.HIGHEST_PROTOCOL)
    with open('saved/accuracies_across_part_ExObs.pkl', 'wb') as f:
        pickle.dump(accuracies_ExObs, f, pickle.HIGHEST_PROTOCOL)
    
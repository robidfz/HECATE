import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
import numpy as np
import networkx as nx
from configparser import ConfigParser
import sys
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import time

def ageGrouping(x):

    if(x>=35 and x<50):
        group=1
    elif(x >= 50 and x < 65):
        group = 2
    else:
        group=3
    return group
def preprocessing(config,df):
    reader = ConfigParser()
    reader.read(config)
    features_initials = reader['FEATURES']['f_list'].split(',')
    additive_features = reader['ADDITIVE FEATURES']['f_list'].split(',')
    target = reader['NETWORK SETTING']['target'].split(',')
    net_number = int(reader['NETWORK SETTING']['number_of_net'])
    setting = reader['SETTING1']['property']
    if (setting == 'true'):
        feature = reader['SETTING1']['value']
        new_feature = reader['SETTING1']['new_feature']
        df[new_feature] = df[feature].apply(ageGrouping)
    setting = reader['SETTING2']['property']
    if (setting == 'true'):
        feature = reader['SETTING2']['value']
        opt1 = reader['SETTING2']['opt1']
        opt2 = reader['SETTING2']['opt2']
        df[feature] = df[feature].map({opt1: 0, opt2: 1})

    return df,net_number,features_initials,additive_features,target

def buildingDataset(df,features_initials, sg):
    features=features_initials.copy()
    features.append(sg)
    data = df[features]
    couples = list()
    for f in features_initials:
        opt = (f,sg)
        couples.append(opt)
    return couples,data
def drawing_net(model,couples,sg,save_path):
    fig, ax = plt.subplots(figsize=(50, 10))
    G = nx.Graph()
    for elem in couples:
        G.add_node(elem[0], color='blue', size=500)
        G.add_node(elem[1], color='red', size=500)
        G.add_edge(elem[0], elem[1])
    pos = nx.spring_layout(G)
    nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_size=20, arrowsize=20, node_color='red', ax=ax)
    plt.savefig(save_path+'BN_'+sg+'.pdf')
    plt.close()

def CDP_estimation(model,data,target,save_path):
    mle = MaximumLikelihoodEstimator(model=model, data=data)
    cdps = mle.estimate_cpd(node=target)
    cdps.to_csv(save_path+target + "_CDP.csv")


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy,predictions


def best_cv(model, x, y):
    accuracy = 0
    X_train_best = []
    y_train_best = []
    X_test_best = []
    y_test_best = []
    kf = KFold(n_splits=5)
    kf.get_n_splits()
    x[np.isnan(x)] = 0
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, np.ravel(y_train, order='C'))
        if (model.score(X_test, y_test) > accuracy):
            accuracy = model.score(X_test, y_test)
            X_train_best = X_train
            y_train_best = y_train
            X_test_best = X_test
            y_test_best = y_test

    rf = model.fit(X_train_best, np.ravel(y_train_best, order='C'))
    return rf, X_test_best, y_test_best






def feature_importance(rf,features,s,savepath):
    feature_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    sns.barplot(x=feature_imp, y=feature_imp.index)

    xlim=np.max(feature_imp)
    plt.xlim(0,xlim)
    plt.xlabel('Feature importance score')
    plt.ylabel('Features')
    plt.title('Display most important features')
    plt.tight_layout()
    plt.savefig(savepath+'Features_importance_'+s+'.pdf')
    plt.close()


def random_forest_validating(rf,x_val,y_val,target,save_path):
    logfile1 = open(save_path+'log_rf.txt', 'a+')
    time1=time.time()
    predictions=rf.predict(x_val)
    errors = abs(predictions - y_val)
    error_mean=np.mean(errors)
    scoring = rf.score(x_val, y_val)
    time2 = time.time()
    elapsed_time = time2 - time1
    logfile1.write('Time needed for prediction and accuracy evaluation ' + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + '\n')
    logfile1.write('ERRORS MEAN OF THE VALIDATION ON THE SECOND DATASET: ' + str(error_mean) + '\n')
    logfile1.write('SCORING  OF THE VALIDATION ON THE SECOND DATASET: ' + str(scoring) + '\n')
    rf_cm = confusion_matrix(y_val, predictions)
    logfile1.write(str(rf_cm)+'\n')

    rf_cm_plt = sns.heatmap(rf_cm.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues")
    plt.xlabel('Actual label')
    plt.ylabel('Predicted label')
    plt.title("Valid")
    plt.savefig(save_path + target + 'confusion_matrix.png')
    plt.close()



def BN_pipeline(df,features_initials, target,save_path):
    couples,data=buildingDataset(df,features_initials,target)
    model = BayesianNetwork(couples)
    drawing_net(model,couples,target,save_path)
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    CDP_estimation(model,data,target,save_path)


def RF_pipeline(df,features_initials, target,save_path):
    x = df[features_initials]
    y = df[target]
    x = x.to_numpy()
    y = np.asarray(y)
    model=RandomForestClassifier(random_state=1, n_jobs=-1, n_estimators=6)
    rf, X_test_best, y_test_best=best_cv(model, x, y)
    random_forest_validating(rf,X_test_best,y_test_best,target,save_path)
    feature_importance(rf, features_initials, target,save_path)

#Tumor type---> Met 0 Primary 1 ExtraHepatic 2
if __name__ == "__main__":
    if len(sys.argv) == 5:
        config = sys.argv[1]
        dataset=sys.argv[2]
        save_path=sys.argv[3]
        model=sys.argv[4]
        df = pd.read_csv(dataset)
        df,net_number,features_initials,additive_features,target = preprocessing(config,df)
        for i in range(net_number):
            features_initials.append(additive_features[i])
            if(model=='BN'):
                BN_pipeline(df, features_initials, target[i], save_path)
            else:
                RF_pipeline(df, features_initials, target[i], save_path)
            features_initials.remove(additive_features[i])
    else:
        print('error')







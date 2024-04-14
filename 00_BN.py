

import pandas as pd
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import networkx as nx
from configparser import ConfigParser
import sys
def ageGrouping(x):

    if(x>=35 and x<50):
        group=1
    elif(x >= 50 and x < 65):
        group = 2
    else:
        group=3
    return group
def preprocessing(config):
    reader = ConfigParser()
    reader.read(config)
    features_initials = reader['FEATURES']['f_list'].split(',')
    property = reader['ADDITIVE FEATURES']['property']
    if (property == 'true'):
        additive_features = reader['ADDITIVE FEATURES']['f_list'].split(',')
    target = reader['NETWORK SETTING']['target'].split(',')
    net_number = int(reader['NETWORK SETTING']['number_of_net'])
    setting = reader['PREPROCESSING SETTING']['property']
    if (setting == 'true'):
        feature = reader['PREPROCESSING SETTING']['value']
        new_feature = reader['PREPROCESSING SETTING']['new_feature']
        df[new_feature] = df[feature].apply(ageGrouping)

        return net_number,features_initials,additive_features,target
def buildingDataset(df,features_initials, sg):
    features=features_initials.copy()
    features.append(sg)
    data = df[features]
    couples = list()
    for f in features_initials:
        opt = (f,sg)
        couples.append(opt)
    return couples,data
def drawing_net(model,features_initials,sg):
    fig, ax = plt.subplots(figsize=(50, 10))
    G = nx.Graph()
    for elem in couples:
        G.add_node(elem[0], color='blue', size=500)
        G.add_node(elem[1], color='red', size=500)
        G.add_edge(elem[0], elem[1])
    pos = nx.spring_layout(G)
    nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_size=20, arrowsize=20, node_color='red', ax=ax)
    plt.savefig('BN_'+sg+'.pdf')
    plt.close()

def CDP_estimation(model,data,target):
    mle = MaximumLikelihoodEstimator(model=model, data=data)
    cdps = mle.estimate_cpd(node=target)
    cdps.to_csv(target + "_CDP.csv")


#Tumor type---> Met 0 Primary 1 ExtraHepatic 2
if __name__ == "__main__":
    if len(sys.argv) == 3:
        config = sys.argv[1]
        dataset=sys.argv[2]
        df = pd.read_csv(dataset)
        net_number,features_initials, additive_features, target= preprocessing(config)
        for i in range(net_number):
            features_initials.append(additive_features[i])
            couples,data=buildingDataset(df,features_initials,target[i])
            model = BayesianNetwork(couples)
            drawing_net(model,features_initials,target[i])
            model.fit(data, estimator=MaximumLikelihoodEstimator)
            CDP_estimation(model,data,target[i])
            features_initials.remove(additive_features[i])
    else:
        print('error')







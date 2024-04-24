import pandas as pd
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import networkx as nx
from configparser import ConfigParser
import sys


def print_roman(number):
    retval = ""
    num = [1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000]
    sym = ["I", "IV", "V", "IX", "X", "XL", "L", "XC", "C", "CD", "D", "CM", "M"]
    ii = 12
    while number:
        div = number // num[ii]
        number %= num[ii]
        while div:
            retval += sym[ii]
            div -= 1
        ii -= 1
    return retval


def age_grouping(x):
    group = 3
    if 35 <= x < 50:
        group = 1
    elif 50 <= x < 65:
        group = 2
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
        df[new_feature] = df[feature].apply(age_grouping)
        return net_number, features_initials, additive_features, target


def buildingDataset(df, features_initials, sg):
    features = features_initials.copy()
    features.append(sg)
    data = df[features]
    couples = list()
    for f in features_initials:
        opt = (f, sg)
        couples.append(opt)
    return couples, data


def drawing_net(model, features_initials, sg):
    fig, ax = plt.subplots(figsize=(50, 10))
    G = nx.Graph()
    for elem in couples:
        G.add_node(elem[0], color='blue', size=500)
        G.add_node(elem[1], color='red', size=500)
        G.add_edge(elem[0], elem[1])
    pos = nx.spring_layout(G)
    nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_size=20, arrowsize=20, node_color='red', ax=ax)
    plt.savefig('BN_' + sg + '.pdf')
    plt.close()


def CDP_estimation(model, data, target):
    mle = MaximumLikelihoodEstimator(model=model, data=data)
    cdps = mle.estimate_cpd(node=target)
    cdps.to_csv(target + "_CDP.csv")


def get_evidence_names(index):
    pre_name = print_roman(index + 1) + " PRE"
    post_name = print_roman(index + 1) + " POST"
    return pre_name, post_name

'''    
    distribution = ve.query(
        variables=["Sex", "Age Group", "Tumor Type", "CT", "RMN", "PET-CT"],
        evidence={yes_variable: 1, no_variable: 0},
        joint=True
    )
'''

def evidence(yes_variable, no_variable, label):
    outfile_name = evidence_pre_name + "_" + evidence_post_name + "." + label + ".distribution"
    distribution = ve.query(
        variables=["PET-CT"],
        evidence={yes_variable: 1, no_variable: 0},
        joint=True
    )
    outfile_handle = open(outfile_name, "w")
    outfile_handle.write(str(distribution))
    outfile_handle.close()


#Tumor type---> Met 0 Primary 1 ExtraHepatic 2
if __name__ == "__main__":
    if len(sys.argv) == 3:
        configfile_name = sys.argv[1]
        dataset_name = sys.argv[2]
        df = pd.read_csv(dataset_name)
        net_number, features_initials, additive_features, target = preprocessing(configfile_name)
        for i in range(net_number):
            features_initials.append(additive_features[i])
            couples, data = buildingDataset(df, features_initials, target[i])
            model = BayesianNetwork(couples)
            drawing_net(model, features_initials, target[i])
            model.fit(data, estimator=MaximumLikelihoodEstimator)
            CDP_estimation(model, data, target[i])
            features_initials.remove(additive_features[i])
            ve = VariableElimination(model)
            evidence_pre_name, evidence_post_name = get_evidence_names(i)
            evidence(evidence_post_name, evidence_pre_name, "falsepositive")
            evidence(evidence_pre_name, evidence_post_name, "falsenegative")
    else:
        print('error')

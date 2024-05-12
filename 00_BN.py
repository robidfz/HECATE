import numpy as np
import plotly.express as px
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import networkx as nx
from configparser import ConfigParser
import sys
import seaborn as sns
import plotly.io as pio

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
    group = 'elders'
    if 35 <= x < 50:
        group = 'youngs'
    elif 50 <= x < 65:
        group = 'middle-aged'
    return group


def preprocessing(config, dataframe):
    reader = ConfigParser()
    reader.read(config)
    features_initials = reader['FEATURES']['f_list'].split(',')
    property = reader['ADDITIVE FEATURES']['property']
    if property == 'true':
        additive_features = reader['ADDITIVE FEATURES']['f_list'].split(',')
    target = reader['NETWORK SETTING']['target'].split(',')
    net_number = int(reader['NETWORK SETTING']['number_of_net'])
    setting = reader['PREPROCESSING SETTING']['property']
    if setting == 'true':
        feature = reader['PREPROCESSING SETTING']['value']
        new_feature = reader['PREPROCESSING SETTING']['new_feature']
        dataframe[new_feature] = dataframe[feature].apply(age_grouping)
    type = reader['ANALYSIS TYPE']['t_list'].split(',')

    return net_number, features_initials, additive_features, target,type


def buildingDataset(df, features_initials, sg):
    features = features_initials.copy()
    features.append(sg)
    data = df[features]
    couples = list()
    for f in features_initials:
        opt = (f, sg)
        couples.append(opt)
    return couples, data


def drawing_net(model, couples, sg):
    fig, ax = plt.subplots(figsize=(12, 10))
    G = nx.Graph()
    for elem in couples:
        G.add_node(elem[0], color='grey', size=500)
        G.add_node(elem[1], color='white', size=1000)
        G.add_edge(elem[0], elem[1])
    pos = nx.spring_layout(G)
    nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_size=16, arrowsize=30, node_color='#D3FBFB',
            ax=ax, font_color="black", edge_color="black")
    latex_code = nx.to_latex(G, pos=pos)
    log_file = open("latex_code.txt", 'a+')
    log_file.write("Latex code for BN of the segment "+sg+"\n")
    log_file.write(latex_code)
    log_file.write("\n\n\n")
    log_file.close()
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


def evidence(pre_variable, post_variable, label, target, analysisengine):
    if label == 'False Positives':
        true_variable = post_variable
        false_variable = pre_variable
    else:
        false_variable = post_variable
        true_variable = pre_variable
    distribution = analysisengine.query(
        variables=[target],
        evidence={true_variable: 'Positive', false_variable: 'Negative'},
        joint=True
    )
    return tuple(distribution.values)


def processnames(states):
    retval = dict(states)
    for k in retval.keys():
        retval[k] = list(map(lambda x: str(x), retval[k]))
    return retval
def building_network(df,features_initials,net_number,targets ):
    models = dict()
    segments = []
    statelabels = None
    for i in range(net_number):
        features_initials.append(additive_features[i])
        couples, data = buildingDataset(df, features_initials, targets[i])
        model = BayesianNetwork(couples)
        drawing_net(model, couples, targets[i])
        model.fit(data, estimator=MaximumLikelihoodEstimator)
        CDP_estimation(model, data, targets[i])
        features_initials.remove(additive_features[i])
        models[i] = model
        if statelabels is None:
            statelabels = processnames(model.states)  # from array to dictionary
        segments.append(print_roman(i + 1))
    segmentnames = list(segments)
    segments = list(map(lambda x: (x + " PRE", x + " POST"), segments))
    return models,segments
def analysis(features_initials,type,models,net_number,segments):
    average_deltas=list()
    for target in features_initials:
        deltas_features = list()
        for analysis in type:
            aposteriori = []
            deltas = []
            apriori = None
            for i in range(net_number):
                model = models[i]
                ve = VariableElimination(model)
                if apriori is None:
                    apriori = ve.query(variables=[target], joint=True).values
                evidence_pre_name, evidence_post_name = segments[i]
                analysisresults = evidence(evidence_pre_name, evidence_post_name, analysis, target, ve)
                aposteriori.append(analysisresults)
                deltas.append(abs(apriori - aposteriori[i]))
            aposteriori = list(map(list, zip(*aposteriori)))
            deltas = list(map(list, zip(*deltas)))
            values_tot = 0
            for values in deltas:
                for v in values:
                    values_tot += v
            values_tot = values_tot / len(deltas)
            deltas_features.append(values_tot)
        average_deltas.append(deltas_features)
    return average_deltas

def drawing_features_impact(average_deltas,type):
    df = pd.DataFrame()
    df['Features'] = features_initials
    for t, v in enumerate(type):
        df[v] = [elem[t] for elem in average_deltas]
    df.sort_values(by=type, inplace=True, ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = df['Features']
    width = 0.4
    plot_title = 'Features Impact'
    title_size = 18
    filename = 'features_impact'
    y = np.arange(len(labels))
    width_list=[y + width / 2, y - width / 2]
    for i,t in enumerate(type):
        first_bar = df[t]
        first_bar_label = t
        ax.barh(width_list[i], first_bar, width, label=first_bar_label)
    title = plt.title(plot_title, fontsize=title_size)
    ax.set_yticklabels(labels)
    ax.legend()
    plt.yticks(np.arange(min(y), max(y) + 1, 1.0))
    plt.savefig(filename + '.pdf')


if __name__ == "__main__":
    if len(sys.argv) == 3:
        configfile_name = sys.argv[1]
        dataset_name = sys.argv[2]
        df = pd.read_csv(dataset_name)
        net_number, features_initials, additive_features, targets,type= preprocessing(configfile_name, df)
        models, segments= building_network(df,features_initials,net_number,targets)
        average_deltas=analysis(features_initials,type,models,net_number,segments)
        drawing_features_impact(average_deltas,type)









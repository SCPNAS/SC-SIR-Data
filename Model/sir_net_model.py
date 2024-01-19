import numpy as np
import future
import networkx as nx
from future.utils import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint
import random
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import pandas as pd
from numpy import loadtxt
import os
import csv
from collections import Counter
import itertools
from sklearn.metrics import mean_squared_error

path = os.getcwd()[:-5] + 'Dataset/'

def status_delta(available_statuses, status, actual_status):
    actual_status_count = {}
    old_status_count = {}
    delta = {}
    for n, v in iteritems(status):
        if v != actual_status[n]:
            delta[n] = actual_status[n]
    for st in list(available_statuses.values()):
        actual_status_count[st] = len(
            [x for x in actual_status if actual_status[x] == st])
        old_status_count[st] = len([x for x in status if status[x] == st])
    _status_delta = {st: actual_status_count[st] - old_status_count[st] for st in actual_status_count}

    return delta, actual_status_count, _status_delta

available_statuses = {
    "Susceptible": 0,
    "Infected": 1,
    "Removed": 2
}

# iteration starts #
def SIR_model(init_status, actual_iteration, num_iteration, beta, gamma, graph):
    status = init_status
    nodes_status = np.zeros(shape=(num_iteration, len(graph)), dtype=np.float32)
    g = graph
    while actual_iteration < num_iteration:
        print("iteration:", actual_iteration, num_iteration)
        for n, s in iteritems(status):
            if s not in available_statuses.values():
                status[n] = 0

        actual_status = {node: nstatus for node, nstatus in iteritems(status)}

        if actual_iteration == 0:
            nodes_status[0] = list(status.values())
            actual_iteration = actual_iteration + 1
            _, _, _status_delta = status_delta(available_statuses, status, actual_status)

        for u in g.nodes:
            u_status = status[u]
            eventp = np.random.random_sample()
            # for the 1-hop neighborhood
            neighbors = list(g.neighbors(u))

            infected_neighbors = [v for v in neighbors if status[v] == 1]

            if u_status == 0:  # susceptible
                if eventp < 1 - (1-beta)**len(infected_neighbors):
                    actual_status[u] = 1  # Infected

            elif u_status == 1:
                if eventp < gamma:
                    actual_status[u] = 2  # Recovered

        _, _, _status_delta = status_delta(available_statuses, status, actual_status)
        status = actual_status

        if actual_iteration == 0:
            print(status)

        nodes_status[actual_iteration] = list(status.values())  # save
        actual_iteration += 1

    return nodes_status

# load data
seed = 1992
county_name = 'albemarle'
FIPS_code = 51003
num_hours = 8
type_net = 'whole'
net = nx.read_gml(path + '/' + str(county_name) + '/' + str(county_name) + '_county_' + str(num_hours) + '_hours_' + type_net + '_net.gml')
net = nx.convert_node_labels_to_integers(net, first_label=0)
triangles = np.load(str(county_name) + '_county_' + str(num_hours) + '_hours_' + type_net + '_contact_network_triangles_list.npz', allow_pickle=True)['arr_0']

us_county_confirmed = (pd.read_csv(path + '/' + 'US_covid19_confirmed.csv')).values
us_county_death = (pd.read_csv(path + '/' + 'US_covid19_deaths.csv')).values
labels = np.where(us_county_confirmed[:,4] == FIPS_code)
confirmed_cases_county = us_county_confirmed[labels].reshape(-1,)
death_cases_county = us_county_death[labels].reshape(-1,)
active_recovered_cases_county = confirmed_cases_county[11:]-death_cases_county[12:]
start_day = 770
study_days = 30
total_pop_county = 113535 # population
y = active_recovered_cases_county[start_day:start_day+study_days]/ total_pop_county

np.random.seed(seed)
num_init_infected = int((y*total_pop_county)[0]/(total_pop_county/len(net.nodes)))
infected_node = (np.random.choice(net.nodes, size=num_init_infected, replace=False))
init_status = {n: 1 if n in infected_node else 0 for n in list(net.nodes)}

# hyperparameters
sir_beta = 0.0001
sir_gamma = 0.01

# modeling
final_res = []
sim_num = 5 # number of simulation
days = study_days
for i in range(sim_num):
    np.random.seed(i)
    res = SIR_model(init_status=init_status, actual_iteration=0, num_iteration=days, beta=sir_beta, gamma=sir_gamma, graph=net)
    tmp_res = np.array([np.unique(np.where((res[0, :]) == 1)).shape[0] if ii==0 else np.unique(np.where((res[:ii, :]) == 1)[1]).shape[0] for ii in range(days)]) / (len(net.nodes()))
    final_res.append(tmp_res)
final_res = np.array(final_res).reshape(sim_num, days)

# evaluation (RMSE)
sir_net_rmse = []
for i in range(sim_num):
        tmp = np.sqrt(mean_squared_error(y, final_res[i,:]))
        sir_net_rmse.append(tmp)
avg_sir_net_rmse = np.mean(sir_net_rmse)
std_sir_net_rmse = np.std(sir_net_rmse)

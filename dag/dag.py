import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

dat = pd.read_csv("BreakingData.csv")
# dat.head()

edges = [
    ('Object', 'Collision'), 
    ('Object', 'Visual_Sensor_Detection'),
    ('Object', 'LIDAR_Sensor'),

    ('LIDAR_Sensor', 'Sensor_Detection'),

    ('Visual_Sensor_Detection', 'Sensor_Detection'),

    ('Light_Dark', 'Visual_Sensor_Detection'),
    ('Light_Dark', 'Sensor_Detection'),

    ('Sensor_Detection', 'Early_Breaking'),

    ('Weather_Visibility', 'Weather_Detection'),
    ('Weather_Visibility', 'Visual_Sensor_Detection'),

    ('Road_Condition', 'Road_Condition_Detection'),
    ('Road_Condition', 'Collision'),

    ('Road_Condition_Detection', 'Early_Breaking'),

    ('Early_Breaking', 'Collision')
]

mod = BayesianModel(edges)

nodes = []
for i in edges:
    nodes.append(i[0])
    nodes.append(i[1])

# Tabular CPDS

for n in ["Road_Condition", "Weather_Visibility", "Light_Dark", "Object"]:
    mod.add_cpds(TabularCPD(variable=n, variable_card=2, values=[[0, 1]]))

mod.add_cpds(TabularCPD(variable='Road_Condition_Detection',
                        variable_card=2, values=[[0, 1], [0, 1]],
                        evidence=['Road_Condition'], evidence_card=[2]))
mod.add_cpds(TabularCPD(variable='Weather_Detection',
                        variable_card=2, values=[[0, 1], [0, 1]],
                        evidence=['Weather_Visibility'], evidence_card=[2]))

mod.add_cpds(TabularCPD(variable='Visual_Sensor_Detection',
                        variable_card=3, values=[[0, 1],
                                                 [0, 1],
                                                 [0, 1]],
                        evidence=['Weather_Visibility',
                                  'Light_Dark',
                                  'Object'],
                        evidence_card=[2, 2]))


    cpd_B, cpd_C, cpd_MO, cpd_W, cpd_M
    )

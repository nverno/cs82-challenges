import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

data = pd.read_csv("BreakingData.csv")
data.head()

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

    ('Weather_Detection', 'Sensor_Detection'),

    ('Road_Condition', 'Road_Condition_Detection'),
    ('Road_Condition', 'Collision'),

    ('Road_Condition_Detection', 'Early_Breaking'),

    ('Early_Breaking', 'Collision')
]
variables = data.columns[1:]
mod = BayesianModel(edges)


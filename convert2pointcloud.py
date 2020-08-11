# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:18:01 2020

@author: sv
"""

#import pandas as pd
from pyntcloud import PyntCloud

cloud = PyntCloud.from_file("testmeshA_landmarks.txt",sep=" ",header=0,names=["x","y","z"])

cloud.to_file("testmeshA_landmarks.ply")

cloud.plot()

#Point_clound = PyntCloud.from_file("PtCloud1.ply")

#Point_clound.plot()


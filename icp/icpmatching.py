# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 09:35:48 2020

@author: sv
"""

import numpy as np
import time
#import icp
from icp import icp3d

import logging, sys

# Constants
#N = 1000                                    # number of random points in the dataset
#num_tests = 1                               # number of test iterations
#dim = 3                                     # number of dimensions of the points
#noise_sigma = .01                           # standard deviation error to be added
#translation = .5                            # max translation of the test set
#rotation = .1                               # max rotation (radians) of the test set

class Icp3DMatching:
    def __init__(self):
        self.N = 1000                                    # number of random points in the dataset
        self.num_tests = 1                               # number of test iterations
        self.dim = 3                                     # number of dimensions of the points
        self.noise_sigma = .01                           # standard deviation error to be added
        self.translation = .5                            # max translation of the test set
        self.rotation = .1                               # max rotation (radians) of the test set
        
    
    
    def test_best_fit(self):
        # Generate a random dataset
        A = np.random.rand(self.N, self.dim)
    
        total_time = 0
    
        for i in range(self.num_tests):
    
            B = np.copy(A)
    
            # Translate
            t = np.random.rand(self.dim)* self.translation
            B += t
    
            # Rotate
            R = icp3d.Icp3D.rotation_matrix(np.random.rand(self.dim), np.random.rand()*self.rotation)
            B = np.dot(R, B.T).T
    
            # Add noise
            B += np.random.randn(self.N, self.dim) * self.noise_sigma
    
            # Find best fit transform
            start = time.time()
            T, R1, t1 = icp3d.best_fit_transform(B, A)
            total_time += time.time() - start
    
            # Make C a homogeneous representation of B
            C = np.ones((self.N, 4))
            C[:,0:3] = B
    
            # Transform C
            C = np.dot(T, C.T).T
    
            assert np.allclose(C[:,0:3], A, atol=6*self.noise_sigma) # T should transform B (or C) to A
            assert np.allclose(-t1, t, atol=6*self.noise_sigma)      # t and t1 should be inverses
            assert np.allclose(R1.T, R, atol=6*self.noise_sigma)     # R and R1 should be inverses
    
        print('best fit time: {:.3}'.format(total_time/self.num_tests))
    
        return
    
    
    
    def test_icp_default(self):
        # Generate a random dataset
        A = np.random.rand(self.N, self.dim)
    
        total_time = 0
    
        for i in range(self.num_tests):
    
            B = np.copy(A)
    
            # Translate
            t = np.random.rand(self.dim)* self.translation
            B += t
    
            # Rotate
            R = icp3d.Icp3D.rotation_matrix(np.random.rand(self.dim), np.random.rand() * self.rotation)
            B = np.dot(R, B.T).T
    
            # Add noise
            B += np.random.randn(self.N, self.dim) * self.noise_sigma
    
            # Shuffle to disrupt correspondence
            np.random.shuffle(B)
    
            # Run ICP
            start = time.time()
            T, distances, iterations = icp3d.Icp3D.icp(B, A, convergence=0.000001)
            total_time += time.time() - start
    
            # Make C a homogeneous representation of B
            C = np.ones((self.N, 4))
            C[:,0:3] = np.copy(B)
    
            # Transform C
            C = np.dot(T, C.T).T
    
            assert np.mean(distances) < 6*self.noise_sigma                   # mean error should be small
            assert np.allclose(T[0:3,0:3].T, R, atol=6*self.noise_sigma)     # T and R should be inverses
            assert np.allclose(-T[0:3,3], t, atol=6*self.noise_sigma)        # T and t should be inverses
    
        print('icp time: {:.3}'.format(total_time/self.num_tests))
    
        return
    
    
    
    def icp3d_matching( filename_obj1, filename_obj2 ):
        # Load Ply A and B
        print('################## Load point cloud : ', filename_obj1 )
        A = icp3d.Icp3D.loadPointCloud(filename_obj1)
        #A = icp3d.Icp3D.loadPointCloud('Pasha_guard_head_landmarks.ply') #Pasha_guard_head_landmarks
        logging.debug("Point Cloud A: %d pts" % len(A))
        
        print('################## Load point cloud : ', filename_obj2 )
        B = icp3d.Icp3D.loadPointCloud(filename_obj2)
        #B = icp3d.Icp3D.loadPointCloud('testmeshA_landmarks.ply')                      #PtCloud1        Pasha_head_landmarks      testmeshA_landmarks
        logging.debug("Point Cloud B: %d pts" % len(B))
    
        #Create a low density point cloud for first pass
        #B_low = icp.decimate_by_sequence(B,20)
        #logging.debug("Point Cloud B: %d pts (After decimate(16))" % len(B_low))
    
        #Create a high density but centrally filtered point cloud for senond pass
        #B = icp.filter_points_by_angle(B,20) #Filter point clouds within 25degrees in Z Axis
        #logging.debug("Point Cloud B : %d pts (After filtering)" % len(B))
    
        total_time = 0
    
        # Run ICP
        print('################# Run ICP')
        start = time.time()
        #T, distances, iterations = icp.icp(B_low, A, convergence=0.00001, standard_deviation_range = 0.0, max_iterations=30)
        T = None
        #T, distances, iterations = icp.icp(B, A, convergence=0.0000001, standard_deviation_range = 0.0, max_iterations=100, init_pose = T , quickconverge = 2)
        T, distances, iterations, mean_error = icp3d.Icp3D.icp(B, A, convergence=0.0000001, standard_deviation_range = 0.0, max_iterations=100, init_pose = T , quickconverge = 2)
    
        total_time += time.time() - start
    
        
        print('icp time: {:.3}'.format(total_time))
    
        return mean_error
    
    
    
    def keywithmaxval(d):
        #version copy from the wolf https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        #a) create a list of the dict's keys and values; 
        #b) return the key with the max value  
         v=list(d.values())
         k=list(d.keys())
         return k[v.index(max(v))]
     
        
    def keywithminval(d):
        #version copy from the wolf https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        #a) create a list of the dict's keys and values; 
        #b) return the key with the max value  
         v=list(d.values())
         k=list(d.keys())
         return k[v.index(min(v))]
        
        
    
    def read_obj_name_file(filename):
        #read obj_name file and save to objname_list
        path = 'assets/obj_name.txt'
        if filename is None:
            filename = path
            
        objname_list = []
        count = 0
        objfile = open(filename, 'r')
        Lines = objfile.readlines() 
        
        for line in Lines:
            objname_list.append(line.strip())
            count = count + 1
            print('line' , count,  ' : ' , line.strip())
        
        objfile.close()
        return objname_list
    
    
    
    
    
    
    
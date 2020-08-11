# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 08:20:28 2020

@author: sv
"""

import logging
import sys  
from math import ceil, cos, radians             #Angle Calculation

import numpy as np                              # General Matrix Computation
from sklearn.neighbors import NearestNeighbors  # Computation of Nearest Neighbour

from plyfile import PlyData, PlyElement         # plyfile library for reading ply point cloud files         
                     

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)
    
def nearest_neighbor(src, dst):
        '''Find the nearest (Euclidean) neighbor in dst for each point in src
        
        Parameters
        ----------
            src: Pxm array of points
            dst: Qxm array of points
    
        Returns
        -------
            distances: P-elements array of Euclidean distances of the nearest neighbor
            indices: P-elements array of dst indices of the nearest neighbor
        '''
        # Asserting src and dst have the same dimension. They can have different point count.
        assert src.shape[1] == dst.shape[1]
    
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()
    
def best_fit_transform(A, B):
        '''
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
        Input:
          A: Nxm numpy array of corresponding points
          B: Nxm numpy array of corresponding points
        Returns:
          T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
          R: mxm rotation matrix
          t: mx1 translation vector
        '''
    
        assert A.shape == B.shape
    
        # get number of dimensions
        m = A.shape[1]
    
        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B
    
        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
    
        # special reflection case
        if np.linalg.det(R) < 0:
           Vt[m-1,:] *= -1
           R = np.dot(Vt.T, U.T)
    
        # translation
        t = centroid_B.T - np.dot(R,centroid_A.T)
    
        # homogeneous transformation
        T = np.identity(m+1)
        T[:m, :m] = R
        T[:m, m] = t
    
        return T, R, t
    
    

class Icp3D:
    def __init__(self):
        self.icp_start = None
        self.icp_end = None
    
    def loadPointCloud(filename ='test.ply'):
        """Load the XYZ location of all points from a ply file
    
        Parameters
        ----------
        filename : str
            path to the ply file (the default is 'test.ply')
    
        Returns
        -------
        numpy array
            Qxm array a list of m-Dimensioned Points, item count reduced from P to Q, where Q = ceil(P/everyNth)
        """
        vertices = PlyData.read(filename)['vertex']   
        #vertices = PlyData.read('test.ply')['vertex']
        assert 'x' in vertices.data.dtype.names
        assert 'y' in vertices.data.dtype.names
        assert 'z' in vertices.data.dtype.names
    
        #Return only the 'x','y','z' values
        return vertices[['x','y','z']].copy().view(np.float32).reshape(-1,3)
    
    
    def rotation_matrix(axis, theta):
        axis = axis/np.sqrt(np.dot(axis, axis))
        a = np.cos(theta/2.)
        b, c, d = -axis*np.sin(theta/2.)
    
        return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                      [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                      [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
    
    
    def format_transformation_matrix (T):
        return ["{},{}".format(i, j) for i, j in T]
    
    
    def angle_between(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
    
                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    
    
    def filter_points_by_angle (A, angle_in_degrees):
        """Reduce the number of items in a point cloud by measuring the poitional vector of each point to the Z axis
        
        Parameters
        ----------
        A : numpy array
            Pxm array for a list with P items of m-Dimensioned Points
        angle_in_degrees : float
            points located within this angle to the Z-Axis is kept 
    
        Returns
        -------
        numpy array
            Qxm array a list of m-Dimensioned Points, item count reduced from P to Q, where Q = ceil(P/everyNth)
        """
        indices = []
        angle = cos(radians(angle_in_degrees))
        for i in range(len(A)):
            if (np.dot([0,0,1],A[i])/np.linalg.norm(A[i]) > angle):
                indices.append(i)
        #This is a very inefficient implementation, someone kwno knows Numpy better should have a better way of dealing with this,
        return A[indices]
    
    
    def decimate_by_sequence (A, everyNth = 2):
        """Reduce the number of points in a point cloud
    
        Parameters
        ----------
        A : numpy array 
            Pxm array for a list with P items of m-Dimensioned Points
        everyNth : int, optional
            Return one point for every x Number of points  (the default is 2, which is to return every other point)
    
        Returns
        -------
        numpy array
            Qxm array a list of m-Dimensioned Points, item count reduced from P to Q, where Q = ceil(P/everyNth)
        """
        return A[range(0,len(A),everyNth)]
    
    
    def icp(A, B, standard_deviation_range = 0.0, init_pose=None, max_iterations=100, convergence=0.001, quickconverge=1, filename="results.txt"):
        '''
        The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
        Input:
            A: Pxm numpy array of source m-Dimensioned points
            B: Qxm numpy array of destination m-Dimensioned point
            standard_deviation_range: If this value is not zero, the outliers (defined by the given standard Deviation) will be ignored.
            init_pose: (m+1)x(m+1) homogeneous transformation
            max_iterations: exit algorithm after max_iterations
            convergence: convergence criteria
            quickconverge: streategy to overapply transformation at each iteration. Value of 2 means two transforamtions are made.
            filename: fileName to save the In-Progress data for preview
        Output:
            T: final homogeneous transformation that maps A on to B
            distances: Euclidean distances (errors) of the nearest neighbor
            i: number of iterations to converge
        '''
    
        # Asserting A and B have the same dimension. They can have different point count.
        assert A.shape[1] == B.shape[1]
    
        # get number of dimensions
        m = A.shape[1]
    
        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m+1,A.shape[0]))
        dst = np.ones((m+1,B.shape[0]))
        src[:m,:] = np.copy(A.T)
        dst[:m,:] = np.copy(B.T)
    
        # apply the initial pose estimation
        if init_pose is not None:
            src = np.dot(init_pose, src)
    
        prev_error = 0
        delta_error = 0
    
        print('##################### open file and wrtite !')
    
        f= open(filename,"w+")
        if standard_deviation_range > 0.0:
            f.write("Outlier purge active, standard deviation range: %f\n", standard_deviation_range)
            logging.debug("Outlier purge active, standard deviation range: %f", standard_deviation_range)
        else:
            f.write("Outlier purge not active.\n")
            logging.debug("Outlier purge not active.")
    
        for i in range(max_iterations):
            f.write("ITERATION:%d\n"%i)
            logging.debug("Iteration: %d "%i)
            #print('################# Iteration', i)
            # find the nearest neighbors between the current source and destination points
            distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
    
            #Compute Mean Error and Standard Deviation
            #print('################# Compute Mean Error and Standard Deviation ')
            mean_error = np.mean(distances)
            stde_error = np.std(distances)
    
            #Ignore distances that are outlers
            if standard_deviation_range > 0.0:
                trimmed_dst_indices = []
                trimmed_src_indices = []
                for j in range(len(indices)):
                    if distances[j] > (mean_error + standard_deviation_range * stde_error):
                        continue
                    if distances[j] < (mean_error - standard_deviation_range * stde_error):
                        continue
                    trimmed_dst_indices.append(indices[j])
                    trimmed_src_indices.append(j)
                    
                # compute the transformation between the selected values in source and nearest destination points
                T,_,_ = best_fit_transform(src[:m,trimmed_src_indices].T, dst[:m,trimmed_dst_indices].T)
                #print('################# Recomputing Mean Error based on selected distances ')
                #Recomputing Mean Error based on selected distances
                mean_error = np.mean(distances[trimmed_src_indices])
                #print('################# mean_error : ', mean_error)
            else:
                # compute the transformation between the source and nearest destination points
                T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)
    
            #Transform the current source
            if quickconverge > 1.0:
                if i > 2:
                    if delta_error > 0.0:
                        quickconverge = quickconverge - 0.4
                        logging.debug("quickconverge: %.2f " % quickconverge)
    
            f.write(" QUICK_CONVERGE: %.1f" % quickconverge)
            f.write("\n")
            for p in range(int(ceil(quickconverge))):
                src = np.dot(T, src)
    
            #Log the current transformation matrix
            currentT,_,_ = best_fit_transform(A, src[:m,:].T)
            f.write(" TRANSFORMATION:")
            f.write(np.array2string(currentT.flatten(),formatter={'float':lambda x: "%.8f" % x},separator=",").replace('\n', ''))
            f.write("\n")
    
            # logging.debug(np.array2string(T.flatten(),separator=","))
    
            # check error to see if convergence cretia is met
            delta_error = (mean_error - prev_error)
            f.write(" MEAN_ERR:%.8f\n" % mean_error)
            f.write(" STD_DEV:%.8f\n" % stde_error)
            f.write(" DELTA_ERROR:%.8f\n" % delta_error)
    
            if np.abs(delta_error) < convergence:
                break
            prev_error = mean_error
        
        f.close() 
        
        # calculate final transformation
        T,_,_ = best_fit_transform(A, src[:m,:].T)
    
        return T, distances, i, mean_error

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
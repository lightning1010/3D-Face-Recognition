# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:42:41 2020

@author: sv
"""

#num_obj = 4
#for idx in range(1,num_obj):
#    print('idx = ', idx)
    
path = 'assets/obj_name.txt'
    
    
objname_list = []
count = 0
objfile = open(path, 'r')
Lines = objfile.readlines() 
        
for line in Lines:
   objname_list.append(line.strip())
   count = count + 1
   print('line' , count,  ' : ' , line.strip())
        
    
print('objname_list : ' , objname_list)


count_obj = 0
for obj in objname_list:
    count_obj = count_obj + 1
    
print('count_obj = ', count_obj)




# now create obj_name file save all obj's dot ply
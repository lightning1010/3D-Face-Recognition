# 3D Face Recognition
import argparse
from parse_config import ConfigParser
import deepmvlm
from utils3d import Utils3D
from icp import icpmatching
import os


def process_icp_obj3d( obj3d_list_agr , name_lm_ply_agr ):
    # input : obj3d_list , name_lm_ply(need prediction) , 
    num_obj = 0
    for obj in obj3d_list_agr:
        num_obj = num_obj + 1
        
    print('################### num_obj = ', num_obj)
    
    newdict = {}
    for i in range(num_obj):
        print('################### i = ', i)
        
        #filename_obj = "assets/obj_" + str(i+1) + '/' + obj3d_list_agr[i]
        #print('################### filename_obj', i+1 , ' : ', filename_obj )
              
        filename_obj = obj3d_list_agr[i]
        print('################### filename_obj', i+1 , ' : ', filename_obj )
        
        mean_error = icpmatching.Icp3DMatching.icp3d_matching(name_lm_ply_agr, filename_obj)
        newdict[filename_obj] = mean_error
        
        print('################### ', filename_obj , ' : ', mean_error )
        print('=============================== the end of iteration : ', i+1 , ' ============================================')
    
    obj_result = icpmatching.Icp3DMatching.keywithminval(newdict)                   #dictionary type
    print('################### obj_result : ', obj_result )
    # output: print obj_result ( name of obj's ply file and mean error : dictionary type)



def process_one_file(config, file_name):
    print('Processing ', file_name)
    name_lm_vtk = os.path.splitext(file_name)[0] + '_landmarks.vtk'
    name_lm_txt = os.path.splitext(file_name)[0] + '_landmarks.txt'
    dm = deepmvlm.DeepMVLM(config)
    landmarks = dm.predict_one_file(file_name)
    dm.write_landmarks_as_vtk_points(landmarks, name_lm_vtk)
    dm.write_landmarks_as_text(landmarks, name_lm_txt)
    dm.write_landmarks_as_ply_for_recognition(name_lm_txt)
    
    name_lm_ply = os.path.splitext(file_name)[0] + '_landmarks.ply'
    
    print('################### test visualise mesh and landmarks ! #################### \n')
    dm.visualise_mesh_and_landmarks(file_name, landmarks)
    obj3d_list = icpmatching.Icp3DMatching.read_obj_name_file(None)     # input is None or path of file
    process_icp_obj3d(obj3d_list, name_lm_ply)





def process_file_list(config, file_name):
    print('Processing filelist ', file_name)
    names = []
    with open(file_name) as f:
        for line in f:
            line = (line.strip("/n")).strip("\n")
            if len(line) > 4:
                names.append(line)
    print('Processing ', len(names), ' meshes')
    dm = deepmvlm.DeepMVLM(config)
    for file_name in names:
        print('Processing ', file_name)
        name_lm_txt = os.path.splitext(file_name)[0] + '_landmarks.txt'
        landmarks = dm.predict_one_file(file_name)
        dm.write_landmarks_as_text(landmarks, name_lm_txt)


def process_files_in_dir(config, dir_name):
    print('Processing files in  ', dir_name)
    names = Utils3D.get_mesh_files_in_dir(dir_name)
    print('Processing ', len(names), ' meshes')
    dm = deepmvlm.DeepMVLM(config)
    for file_name in names:
        print('Processing ', file_name)
        name_lm_txt = os.path.splitext(file_name)[0] + '_landmarks.txt'
        landmarks = dm.predict_one_file(file_name)
        dm.write_landmarks_as_text(landmarks, name_lm_txt)


def main(config):
    name = str(config.name)
    if name.lower().endswith(('.obj', '.wrl', '.vtk', '.ply', '.stl')) and os.path.isfile(name):
        process_one_file(config, name)
    elif name.lower().endswith('.txt') and os.path.isfile(name):
        process_file_list(config, name)
    elif os.path.isdir(name):
        process_files_in_dir(config, name)
    else:
        print('Cannot process (not a mesh file, a filelist (.txt) or a directory)', name)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Deep-MVLM')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-n', '--name', default=None, type=str,
                      help='name of file, filelist (.txt) or directory to be processed')

    global_config = ConfigParser(args)
    main(global_config)

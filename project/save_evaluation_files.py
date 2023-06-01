###
import numpy as np 
import matplotlib.pyplot as plt
# from rich import print
# from rich.progress import track
# from rich.console import Console
# from icecream import ic
import os
###
from PIL import Image

def export_solutions(image_index, solutions, path = "data_project", group_id = "00"):
    """
    Wrapper funciton to load image and save solution

    solutions :
        image_index : index of the image to solve

        list with the following items
        solutions [0] = segmented mask of the puzzle (matrix of 2000x2000 dimentions) , 0 for background, 1 for puzzle piece 
        
        solutions [1] = matrix containing the features of the puzzles.  if there were  N pieces in the puzzle, and you extracted M Features per puzzle piece, then the feature map should be of size N x M

        solutions [2] =  list of lists of images, each list of images is a cluster of puzzle pieces. (it includes outliers as the last elementof the list)
                        If there are k clusters, then the list should have k elements, each element is a list
                        e.g. 

                    solution [2] [0] =  [cluster0_piece0 , cluster0_piece1 ...]
                    solution [2] [1] =  [cluster1_piece0 , cluster1_piece1 ...]
                    solution [2] [2] =  [cluster2_piece0 , cluster2_piece1 ...]
                    ....
                    solution [2] [k]   =  [clusterk_piece0 , clusterk_piece1 ... ]
                    solution [2] [k+1] = [ outlier_piece0, outlier_piece1 ...]
                        
    solutions [3] = list of images containing the puzzles
                    e.g.
                    solution [3] [0] =  solved_puzzle0 (image of 128*3 x 128*4)
                    solution [3] [1] =  solved_puzzle1 (image of 128*4 x 128*4)
                    solution [3] [2] =  solved_puzzle2 (image of 128*3 x 128*3)
                    ....



        folder : folder where the image is located, the day of the exam it will be "test"
        path : path to the folder where the image is located

        group_id : group id of the team
            
    ----------
    image:
        index number of the dataset

    Returns
    """
    
    saving_path = os.path.join(path , "solutions_group_" + str(group_id) )
    if not os.path.isdir(saving_path):
        os.mkdir(saving_path)

    print("saving solutions in folder: " , saving_path)

   
    ## call functions to solve image_loaded
    save_mask           (image_index , solutions[0] , saving_path)
    save_feature_map    (image_index , solutions[1] , saving_path)
    save_cluster        (image_index , solutions[2] , saving_path)
    save_solved_puzzles (image_index , solutions[3] , saving_path)

    
    return None

def save_mask(image_index , solution, saving_path):
    
    filename = os.path.join(saving_path, f"mask_{str(image_index).zfill(2)}.png")
    if solution.shape[0] != 2000 or solution.shape[1] != 2000:
        print("error in mask:  shape of image" , solution.shape)
        return
    if np.max(solution) ==1:
        solution = solution*255
    solution = np.array(solution , dtype = np.uint8)
    Image.fromarray(solution).save(filename)

def save_feature_map(image_index , solution, saving_path):
    filename = os.path.join(saving_path, f"feature_map_{str(image_index).zfill(2)}.txt")
    np.savetxt(filename , solution)

    #min max into 0 ,255 interval
    solution = (solution - np.min(solution)) / (np.max(solution) - np.min(solution))
    solution = np.array(solution*255 , dtype = np.uint8)
    filename = filename.replace(".txt" , ".png")
    Image.fromarray(solution).save(filename)
    
def save_cluster(image_index , solution, saving_path):

    
    filename = os.path.join(saving_path, f"cluster_images_{str(image_index).zfill(2)}.png")

    n_clusters = len(solution)
    len_clusters = [len(cluster) for cluster in solution]


    xlen = n_clusters*128
    ylen = np.max(len_clusters)*128
    
    
    whole_image = np.zeros((xlen ,ylen , 3) , dtype = np.uint8)

    for i in range(n_clusters):
        for j in range(len_clusters[i]):
            if solution[i][j].shape[0] != 128 or solution[i][j].shape[1] != 128:
                print("error in shape of image" , solution[i][j].shape)
                return
            whole_image[i*128:(i+1)*128 , j*128:(j+1)*128 , :] = solution[i][j]
    
    Image.fromarray(whole_image).save(filename)

def save_solved_puzzles(image_index , solution, saving_path):

    n_solutions = len(solution)

    for i , sol in enumerate(solution):
        print(sol.shape)
        sol = np.array(sol , dtype = np.uint8)
        filename = os.path.join(saving_path, f"solved_puzzle_{str(image_index).zfill(2)}_{str(i).zfill(2)}.png")
        Image.fromarray(sol).save(filename)



if __name__ == '__main__':
    #example of how to use the function

    ## random data
    mask = np.random.randint(0,2,(2000,2000))
    feature_map = np.random.rand(30, 200)
    cluster = [[np.zeros((128,128,3)) + np.random.randint(i*100 , (i+1)*80, size=3) for _ in range(np.random.randint(10,20))] for i in range(3)]

    ## append the outliers too!
    outliers =[np.zeros((128,128,3)) + np.random.randint(0,254 , size =3 ) for _ in range(3)]
    cluster.append(outliers)

    solved_puzzle = [np.zeros((128*3,128*4,3)) + np.random.randint(0,254, size=3) for _ in range(3)]

    solution = [mask, feature_map, cluster, solved_puzzle]

    #saving solution for image 1
    export_solutions(1,  solution, path = "data_project", group_id = "00")



import os
import time
import numpy as np
from ase.io.trajectory import Trajectory
from graph.convertor import GraphConvertor

class Traj2Graph():
    def __init__(
        self,
        cutoff:float=3.75,
        pmin:float=2,
        pmax:float=10,
        num_exponents:int=6,  ## also called M
        supercell_matrix=None,  ## [3,2], default to be [[-1,2],[-1,2],[-1,2]]
    ):
        self.convertor=GraphConvertor(cutoff, pmin, pmax, num_exponents, supercell_matrix)
        
    def load(
        self,
        path,
        length=None,
        shuffle=True,
    ):
        files=os.listdir(path)
        files.sort()
        if shuffle:
            files=np.random.permutation(files).tolist()
        start_time=time.time()
        graph_list=[]
        for file in files:
            # print(file+" is loading...")
            if file.endswith(".traj"):
                traj=Trajectory(os.path.join(path, file))
                for atoms in traj:
                    graph_list.append(self.convertor.convert(atoms))
                    if length is not None and len(graph_list)==length:
                        end_time=time.time()
                        return graph_list, end_time-start_time
                    else:
                        continue
        end_time=time.time()
        return graph_list, end_time-start_time
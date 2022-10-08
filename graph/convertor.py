import numpy as np
import ase
import torch
from torch_geometric.data import Data
from feature.eddp_feature import triplet_feature, twobody_feature

class GraphConvertor():
    def __init__(
        self,
        cutoff:float=3.75,
        pmin:float=2,
        pmax:float=10,
        num_exponents:int=6,  ## also called M
        supercell_matrix=None,  ## [3,2], default to be [[-1,2],[-1,2],[-1,2]]
    ):
        self.cutoff = cutoff
        self.exponents=self.get_exponents(pmin, pmax, num_exponents)
        if supercell_matrix is None:
            self.sup_x = np.array([-1, 2], dtype=np.int32)
            self.sup_y = np.array([-1, 2], dtype=np.int32)
            self.sup_z = np.array([-1, 2], dtype=np.int32)
        else:
            self.sup_x=supercell_matrix[0]
            self.sup_y=supercell_matrix[1]
            self.sup_z=supercell_matrix[2]
    
    def convert(
        self,
        atoms:ase.Atoms,  ## ase Atoms  
    ):
        args={}
        positions=np.array(atoms.get_positions(),dtype=np.float)
        cell=np.array(atoms.get_cell(),dtype=np.float)
        triplet_f = triplet_feature(positions, self.exponents, self.exponents, self.cutoff, cell, self.sup_x, self.sup_y, self.sup_z)
        triplet_f=np.array(triplet_f,dtype=np.float)
        twobody_f = twobody_feature(positions, self.exponents, self.cutoff, cell, self.sup_x, self.sup_y, self.sup_z)
        twobody_f=np.array(twobody_f,dtype=np.float)
        onebody_f = np.array(atoms.get_atomic_numbers()).reshape(-1,1)
        feature=np.concatenate([onebody_f, triplet_f, twobody_f], axis=1)  ## [num_atoms, 1+M+M^2]
        args["num_atoms"]=len(atoms)
        args["num_nodes"]=len(atoms)
        args["positions"]=torch.from_numpy(positions).float()
        args["cell"]=torch.from_numpy(cell).float()
        args["feature"]=torch.from_numpy(feature).float()
        args["energy"]=atoms.get_potential_energy()
        args["forces"]=torch.from_numpy(np.array(atoms.get_forces())).float()
        args["stress"]=torch.from_numpy(np.array(atoms.get_stress())).float()
        graph=Data(**args)
        return graph
        
        
    def get_exponents(
        self,
        pmin=2,
        pmax=10,
        num_exponnets=6,
    ):
        beta=(pmax/pmin)**(1/(num_exponnets-1))
        exponents=np.zeros(num_exponnets,dtype=np.float)
        for i in range(num_exponnets):
            exponents[i]=pmin*beta**i
        return exponents
import numpy as np

class Normalizer_Per_Atom():
    """
        normalize property per atoms
    """
    def __init__(
        self,
    ):
        pass
    
    def normalize(
        self,
        datalist:list, ## list of graphs
        mean=None,
        std=None,
    ):
        if mean is None and std is None:
            energies=[]
            for graph in datalist:
                energy=graph.energy
                num_atom=graph.num_atoms
                energy_per_atom=energy/num_atom
                for i in range(num_atom):
                    energies.append(energy_per_atom)
            mean=np.mean(energies)
            std=np.std(energies)
        for graph in datalist:
            energy=graph.energy
            energy_per_atom=energy/graph.num_atoms
            graph.energy=(energy_per_atom-mean)/std*graph.num_atoms
        return mean,std,datalist
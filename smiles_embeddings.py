from rdkit import Chem
import numpy as np
import torch.nn.functional as F
import torch

MAX_MOLECULE_SIZE = 28
BOND_TOLERANCE = 1 / 1.1
ZERO_TOLERANCE = 0.20
CARBON_ATOMIC_NUMBER = 6

class SmilesEmbeddings:
    def __init__(self, max_molecule_size=MAX_MOLECULE_SIZE, 
                 bond_tolerance=BOND_TOLERANCE,
                 zero_tolerance=ZERO_TOLERANCE):
        self.max_molecule_size = max_molecule_size
        self.bond_tolerance = bond_tolerance
        self.zero_tolerance = zero_tolerance

    def embed_smiles(self, molecule_smiles : str):
        mol = Chem.MolFromSmiles(molecule_smiles)
        try:
            distance_matrix = Chem.GetDistanceMatrix(mol, 
                                                     useBO = True, 
                                                     useAtomWts = True)
        except:
            print(molecule_smiles)
        distance_matrix = torch.tensor(distance_matrix, dtype=torch.float32)
        width = distance_matrix.shape[0]
        l_pad = int((self.max_molecule_size - width)/2)
        r_pad = self.max_molecule_size - width - l_pad
        padding = (l_pad, r_pad, l_pad, r_pad)
        padded_distance_matrix = F.pad(distance_matrix, 
                                       padding, 
                                       value=0)
        return padded_distance_matrix

    def unembed_smiles(self, padded_distance_matrix : torch.Tensor):
        return self.dist_to_mol(self.unpad(padded_distance_matrix))
    
    def unpad(self, padded_distance_matrix : torch.Tensor):
        padded_distance_matrix = padded_distance_matrix.detach()
        padded_distance_matrix = torch.where(padded_distance_matrix < 0.20, 
                                             0, 
                                             padded_distance_matrix)
        sum = torch.sum(padded_distance_matrix, 
                        dim=0)
        width = sum[sum.nonzero()].shape[0]
        start = int((self.max_molecule_size - width)/2)
        end = width + start
        unpad = padded_distance_matrix[start:end, start:end]
        return unpad
    
    def dist_to_mol(self, distance_matrix : torch.Tensor):
        distance_matrix = distance_matrix.detach()
        atoms = torch.diagonal(distance_matrix, 0)
        atom_numbers = self.retrieve_atomic_number(atoms)
        
        mol = Chem.RWMol()
        node_to_idx = {}
        for i in range(len(atom_numbers)):
            molIdx = mol.AddAtom(Chem.Atom(atom_numbers[i]))
            node_to_idx[i] = molIdx

        distance_matrix = 1/distance_matrix
        distance_matrix = torch.where(distance_matrix < self.bond_tolerance, 
                                      0, 
                                      distance_matrix)
        distance_matrix = torch.round(distance_matrix)

        for ix, row in enumerate(distance_matrix):
            for iy, bond in enumerate(row):

                # only traverse half the matrix
                if iy <= ix:
                    continue

                # add relevant bond type (there are many more of these)
                if bond == 0:
                    continue
                elif bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                    mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                    mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                    mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.QUADRUPLE
                    mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
        return mol


    def retrieve_atomic_number(self, atom : torch.Tensor):
        return [int(e) 
                for e in 
                np.round(CARBON_ATOMIC_NUMBER/atom.detach().numpy()
                         ).astype(int).tolist()]
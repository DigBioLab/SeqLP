import glob
import Bio.PDB
import pandas as pd
import rmsd
import numpy as np 

class MakeTable:
    def __init__(self, path_to_files) -> None:
        self.all_files = self.collect_files(path_to_files)
        self.table = pd.DataFrame(columns=["filepath", "filename", "sequence", "coordinates"])
    
    @staticmethod
    def collect_files(path_to_files:str):
        files  = glob.glob(path_to_files + "/*.pdb")
        if len(files) == 0:
            raise FileNotFoundError("No PDB files found in the given directory.")
        return files
    
    
    def is_single_chain(self, pdb_file):
        parser = Bio.PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('pdb', pdb_file)
        return len(list(structure.get_chains())) == 1

    def get_sequence(self, pdb_file):
        parser = Bio.PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('pdb', pdb_file)
        pp = Bio.PDB.PPBuilder()
        for pp in pp.build_peptides(structure):
            return pp.get_sequence()

    def get_coordinates(self, pdb_file):
        parser = Bio.PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('pdb', pdb_file)
        ca_atoms = [atom for atom in structure.get_atoms() if atom.get_name() == 'CA']
        coordinates = np.array([atom.coord for atom in ca_atoms])
        return coordinates
                    
    @staticmethod
    def calculate_rmsd(target_coords, ref_coords):
        """Calculate the RMSD between two sets of coordinates."""
        rmsd_value = rmsd.kabsch_rmsd(target_coords, ref_coords)

        return rmsd_value

    def process_files(self):
        data = []
        for file in self.all_files:
            if self.is_single_chain(file):
                try:
                    sequence = self.get_sequence(file)
                    if file == self.all_files[0]:
                        ref_coords = self.get_coordinates(file)
                    else:
                        pass
                    
                    coordinates = self.get_coordinates(file)
                    filename = file.split('/')[-1].split(".")[0]      
                    delta_rmsd = self.calculate_rmsd(coordinates, ref_coords)
                    self.table = data.append({"filepath": file, "filename": filename, "sequence": sequence, "coordinates": coordinates, "delta_rmsd": delta_rmsd})
                except:
                    print(f"No success with {file}")
        self.table = pd.DataFrame(data)
        
        

    
Tabu = MakeTable(r"/zhome/20/8/175218/NLP_train/validation/H_Free_Martin")
Tabu.process_files()
print(Tabu.table.head())
print(Tabu.table.shape)
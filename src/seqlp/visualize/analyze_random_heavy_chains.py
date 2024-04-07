import glob
import Bio.PDB
import pandas as pd

class MakeTable:
    def __init__(self, path_to_files) -> None:
        self.all_files = self.collect_files(path_to_files)
        self.table = None
    
    def collect_files(path_to_files:str):
        return glob.glob(path_to_files + "*.pdb")
    
    
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
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        return atom.get_coord()

    def process_files(self):
        for file in self.all_files:
            if self.is_single_chain(file):
                sequence = self.get_sequence(file)
                coordinates = self.get_coordinates(file)
                filename = file.split('/')[-1]
                self.table = self.table.append({"filepath": file, "filename": filename, "sequence": sequence, "coordinates": coordinates}, ignore_index=True)
    
    

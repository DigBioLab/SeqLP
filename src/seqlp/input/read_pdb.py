from Bio import PDB
from Bio.PDB.Polypeptide import three_to_one
class ReadPDB:
    def __init__(self, pdb_file:str) -> None:
        self.structure = self.parse(pdb_file)
        
    @staticmethod
    def parse(pdb_path:str):
        parser = PDB.PDBParser()
        structure = parser.get_structure('PDB_ID', pdb_path)
        return structure
    
    def _get_sequences_per_chain(self) -> dict:
        chain_sequences = {}

        # Iterate through each model (assuming you just want the first model)
        model = structure[0]

        # Iterate through each chain in the model
        for chain in model:
            sequence = []
            # Iterate through each residue in the chain
            for residue in chain:
                if PDB.is_aa(residue, standard=True):  # Check if the residue is an amino acid (and not water or other heteroatom)
                    try:
                        sequence.append(three_to_one(residue.resname))  # Convert three-letter codes to one-letter
                    except KeyError:
                        continue  # Skip residues that aren't standard amino acids (like selenomethionine)
            # Join the sequence list into a string and add to dictionary with chain ID as key
            chain_sequences[chain.id] = ''.join(sequence)
        return chain_sequences
                
pdb_path = r"c:\Users\nilsh\Downloads\7eow.pdb"
PDBReader = ReadPDB(pdb_path)
chain_sequences = PDBReader._get_sequences_per_chain()
print(chain_sequences)


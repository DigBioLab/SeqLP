
class ParseFasta:

    @staticmethod
    def read_fasta(file_path):
        """
        Reads a FASTA file and returns a dictionary with headers as keys and sequences as values.

        :param file_path: Path to the FASTA file.
        :return: Dictionary with headers and sequences.
        """
        fasta_dict = {}
        header = None
        sequence = []

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith(">"):
                    if header:
                        fasta_dict[header] = ''.join(sequence)
                    header = line[1:]  # Remove '>'
                    sequence = []
                else:
                    sequence.append(line)

            if header:
                fasta_dict[header] = ''.join(sequence)

        return fasta_dict
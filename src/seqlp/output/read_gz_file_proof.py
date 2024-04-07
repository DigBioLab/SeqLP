import gzip

class ReadGZ:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_gz_file(self):
        with gzip.open(self.file_path, 'rb') as f:
            file_content = f.read()
        return file_content

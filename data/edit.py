import gzip
import re
import sys

def replace_in_gz(input_path, output_path):
    with gzip.open(input_path, 'rt', encoding='utf-8') as f_in, \
         gzip.open(output_path, 'wt', encoding='utf-8') as f_out:
        content = f_in.read()
        content = content.replace('\'Valle d"\'Aosta / Vallée d"\'Aoste\'', "Valle d'Aosta / Vallée d'Aoste")
        f_out.write(content)

if __name__ == "__main__":
    replace_in_gz(sys.argv[1], sys.argv[2])

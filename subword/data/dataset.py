# !/usr/bin/env python3

import os
from glob import glob

'''
生成'../../data/example/*_origin.txt', 参考example.md
'''
input_prefix = '../../data/example/*_origin.txt'
outfile = './example.txt'

def merge_files_to_one_file(input_prefix, outfile)
    file_list = glob(input_prefix)
    contents = []
    for file in file_list:
        with open(file, 'r', encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or line == '':
                    continue
                contents.append(line)

    with open(outfile, 'w', encoding='utf8') as f:
        f.write('\n'.join(contents))

if __name__ == '__main__':
    merge_files_to_one_file(input_prefix, outfile)


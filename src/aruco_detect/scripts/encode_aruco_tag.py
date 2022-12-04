#! /usr/bin/env python
from __future__ import print_function

import csv
import sys
import os
import numpy as np

# This encoding file only works for 5x5 tags.
GRID_SIZE = (5, 5)

def main():
    tag_defs, file_names = parse_input()

    print('Processing {} tag definitions'.format(len(tag_defs)))

    encodings = [create_encodings(definition) for definition in tag_defs]

    print(create_cpp_string(encodings, file_names))


def parse_input():
    if len(sys.argv) < 2:
        print_useage()

    tag_defs = []

    file_names = sys.argv[1:]

    for arg in file_names:
        try:
            tag = process_definition_file(arg)
            tag_defs.append(tag)
        except ValueError:
            file_names.remove(arg)
            print('Could not process {}'.format(arg))

    print(file_names)
    return tag_defs, file_names


def print_useage():
    usage_string = 'Usage: encode_aruco_tag.py <tag definition files>'
    print(usage_string)
    sys.exit(1)


def process_definition_file(file_name):
    if not os.path.exists(file_name):
        raise ValueError

    grid = np.genfromtxt(file_name, delimiter=',', dtype='uint8')

    if grid.shape != GRID_SIZE:
        raise ValueError

    return grid


def create_encodings(definition):
    def process_byte(byte):
        return int(sum([num * 2 ** i for i, num in enumerate(reversed(byte))]))

    def encode_rotation(tag):
        tag = tag.reshape(-1)
        return [
            process_byte(tag[0:8]),
            process_byte(tag[8:16]),
            process_byte(tag[16:24]),
            tag[24]
        ]

    return [encode_rotation(np.rot90(definition, n)) for n in range(4)]


def create_cpp_string(encodings, file_names):
    cpp_string = '{{\n'
    for encoding, name in zip(encodings, file_names):
        cpp_string += '    //' + name[:-4] + '\n'
        for rotation in encoding:
            cpp_string += '    {' + ','.join([str(num) for num in rotation]) + '},\n'
        cpp_string += '},{\n'

    cpp_string = cpp_string[:-4] + '}};'

    return cpp_string


if __name__ == "__main__":
    main()

import argparse

from bag.io.file import read_yaml
from bag3_testbenches.measurement.comparator.tran import PWLFDStreamGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('specs', help='YAML specs')
    args = parser.parse_args()

    gen = PWLFDStreamGenerator(read_yaml(args.specs))
    gen.generate()

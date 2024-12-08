#! /usr/bin/env python

import sys

from equitrain import get_args_parser_preprocess
from equitrain import preprocess

# %%

def main():

    parser = get_args_parser_preprocess()

    try:
        preprocess(parser.parse_args())
    except ValueError as v:
        print(v, file=sys.stderr)
        exit(1)

# %%
if __name__ == "__main__":
    main()

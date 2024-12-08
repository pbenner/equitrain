#! /usr/bin/env python

import sys

from equitrain import get_args_parser_train, ArgumentError
from equitrain import train

# %%

def main():

    parser = get_args_parser_train()

    try:
        train(parser.parse_args())
    except ArgumentError as v:
        print(v, file=sys.stderr)
        exit(1)

# %%
if __name__ == "__main__":
    main()

#! /usr/bin/env python
# %%

from equitrain import get_args_parser_train
from equitrain import train

# %%

def main():

    parser = get_args_parser_train()

    train(parser.parse_args())

# %%
if __name__ == "__main__":
    main()

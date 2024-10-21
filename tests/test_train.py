# %%
        
from equitrain import get_args_parser_train
from equitrain import train

# %%

def main():

    r = 4.5

    args = get_args_parser_train().parse_args()

    args.train_file      = f'tests/data/train.h5'
    args.valid_file      = f'tests/data/valid.h5'
    args.statistics_file = f'tests/data/statistics.json'
    args.output_dir      = 'tests/result'

    args.epochs     = 10
    args.batch_size = 64
    args.lr         = 0.01

    train(args)

# %%
if __name__ == "__main__":
    main()

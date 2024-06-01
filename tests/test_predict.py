# %%
        
from equitrain import get_args_parser_predict
from equitrain import predict

# %%

def main():

    r = 4.5

    args = get_args_parser_predict().parse_args()

    args.load_checkpoint_model = f'result/best_val_epochs@1_e@255427.1345/pytorch_model.bin'

    args.predict_file    = f'data/valid.h5'
    args.statistics_file = f'data/statistics.json'

    args.batch_size = 5

    energy_pred, forces_pred, stress_pred = predict(args)

    print(energy_pred)
    print(forces_pred)
    print(stress_pred)

# %%
if __name__ == "__main__":
    main()

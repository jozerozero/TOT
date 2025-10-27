import argparse
import os



parser = argparse.ArgumentParser()

parser.add_argument('-models', type=str, nargs='+', required=True)
parser.add_argument('-size', type=int, default=1)
parser.add_argument('-len', type=int, nargs='+')
parser.add_argument('-dataset', type=str, nargs='+')
parser.add_argument('-seed', type=int, nargs='+')
parser.add_argument('-device', default=1, type=int)
# parser.add_argument('-is_uni', default=False, action='store_true')
args = parser.parse_args()
datalist = args.dataset
file_name = 'nsts'
comand_list = []
type_list = ['type1']
seed_list = args.seed
pred_len_list = args.len
seq_len_list = [60]
for seed in seed_list:
    for method in args.models:
        for dropout in [0]:
            for depth in [10]:
                for hidden_dim in [ 512]:
                    for hidden_layers_enc in [ 2]:
                        for hidden_layers_dec in [ 2]:
                            for tau in [0.75]:
                                for seq_len in seq_len_list:
                                    for lr in [0.003]:
                                        for data in datalist:
                                            for pred_len in  pred_len_list:
                                                for type in type_list:
                                                    command = f'python ./main.py  --dropout {dropout} --method {method} --root_path ./data/ --n_inner 1 --test_bsz 1 ' \
                                                            f'--data {data} --features M --seq_len {seq_len} --label_len 0 --pred_len {pred_len} --des "Exp" --itr 1 ' \
                                                              f'   --train_epochs 24 --learning_rate {lr} --online_learning "full" --sleep_interval 1 ' \
                                                            f'--sleep_epochs 20  --batch_size 6 --depth {depth} --hidden_dim {hidden_dim}' \
                                                              f' --offline_adjust 0 --seed {seed} --gpu {args.device} --hidden_layers_enc {hidden_layers_enc} --hidden_layers_dec {hidden_layers_dec} --tau {tau}  --patience 3'
                                                    comand_list.append(command)

i = 0
while i + args.size <= len(comand_list):
    new_comand = "".join(f"{comand_list[i + j]}  &  " for j in range(args.size)).rstrip().rstrip('&')
    os.system(new_comand)
    # print(new_comand)
    i = i + args.size

if i < len(comand_list):
    new_comand = "".join(f"{comand_list[i + j]}  &  " for j in range(len(comand_list) - i)).rstrip().rstrip('&')
    os.system(new_comand)
    # print(new_comand)

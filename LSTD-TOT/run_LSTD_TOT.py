import argparse
import os

parser = argparse.ArgumentParser()

# parser.add_argument('-methods', type=str, nargs='+', required=True)
parser.add_argument('-size', type=int, default=1)
parser.add_argument('-len', type=int, nargs='+')
parser.add_argument('-dataset', type=str, nargs='+')
parser.add_argument('-seed', type=int, nargs='+', default=[2023])
parser.add_argument('-device', default=0, type=int)
parser.add_argument('-model', default='LSTD_TOT', type=str)
# parser.add_argument('-is_uni', default=False, action='store_true')
args = parser.parse_args()



file_name = args.model
comand_list = []
type_list = ['type1']
seed_list = args.seed
pred_len_list = args.len
for seed in seed_list:
    for data in datalist:
        for pred_len in pred_len_list:
            for type in type_list:
                command = f'python ./main.py  {d[data][pred_len]}' \
                          f' --seed {seed} --gpu {args.device} '
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

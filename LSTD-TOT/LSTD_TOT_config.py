d = {
    'Traffic': {
        1: ' --method LSTD_TOT --root_path ./data/ --data Traffic --n_inner 1 --test_bsz 1 --features M --seq_len 60 --label_len 0 '
           ' --des "Exp" --itr 1 --train_epochs 6  --online_learning "full" --L1_weight 0.001 --dropout 0'
           ' --L2_weight 0.001  --zd_kl_weight 1e-5  --zc_kl_weight 1e-5 --sparsity_weight 1e-5 --pred_len 1 '
           ' --depth 10 --hidden_dim 446 --hidden_layers 1 --tau 0.7 --batch_size 16 --learning_rate 0.003 --mode time --x_dim 9 ',
        24: ' --method LSTD_TOT --root_path ./data/ --data Traffic --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
            ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0 '
            ' --L2_weight 0.001  --zd_kl_weight 1e-5  --zc_kl_weight 1e-5 --sparsity_weight 1e-5 --pred_len 24 '
            ' --depth 9 --hidden_dim 512 --hidden_layers 2 --tau 0.75 --batch_size 16 --learning_rate 0.003 --mode time --x_dim 7 ',
        48: ' --method LSTD_TOT --root_path ./data/ --data Traffic --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
            ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0 '
            ' --L2_weight 0.001  --zd_kl_weight 1e-5  --zc_kl_weight 1e-5 --sparsity_weight 1e-5 --pred_len 48 '
            ' --depth 10 --hidden_dim 512 --hidden_layers 2 --tau 0.7 --batch_size 8 --learning_rate 0.003 --mode time --x_dim 10 '
    },

    'Exchange': {
        1: ' --method LSTD_TOT --root_path ./data/ --data Exchange --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
           ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
           ' --L2_weight 0.001  --zd_kl_weight 1e-5  --zc_kl_weight 1e-5 --sparsity_weight 1e-5 --pred_len 1 '
           ' --depth 10 --hidden_dim 448 --hidden_layers 1 --tau 0.7 --batch_size 4 --learning_rate 0.003 --mode time  --x_dim 9 ',
        24: ' --method LSTD_TOT --root_path ./data/ --data Exchange --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
            ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
            ' --L2_weight 0.001  --zd_kl_weight 1e-5  --zc_kl_weight 1e-5 --sparsity_weight 1e-5 --pred_len 24 '
            ' --depth 10 --hidden_dim 512 --hidden_layers 1 --tau 0.75 --batch_size 8 --learning_rate 0.003 --mode time  --x_dim 10 ',
        48: ' --method LSTD_TOT --root_path ./data/ --data Exchange --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
            ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
            ' --L2_weight 0.001  --zd_kl_weight 1e-5  --zc_kl_weight 1e-5 --sparsity_weight 1e-5 --pred_len 48 '
            ' --depth 9 --hidden_dim 448 --hidden_layers 1 --tau 0.75 --batch_size 8 --learning_rate 0.003 --mode time --x_dim 7 '
    },

    'WTH': {
        1: ' --method LSTD_TOT --root_path ./data/ --data WTH --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
           ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
           ' --L2_weight 0.001  --zd_kl_weight 1e-5  --zc_kl_weight 1e-5 --sparsity_weight 1e-5 --pred_len 1 '
           ' --depth 9 --hidden_dim 500 --hidden_layers 1 --tau 0.75 --batch_size 8 --learning_rate 0.003 --mode time --x_dim 4 ',
        24: ' --method LSTD_TOT --root_path ./data/ --data WTH --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
            ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
            ' --L2_weight 0.001  --zd_kl_weight 1e-5  --zc_kl_weight 1e-5 --sparsity_weight 1e-5 --pred_len 24 '
            ' --depth 9 --hidden_dim 256 --hidden_layers 1 --tau 0.7 --batch_size 8 --learning_rate 0.002 --mode var  --x_dim 10 ',
        48: ' --method LSTD_TOT --root_path ./data/ --data WTH --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
            ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0 '
            ' --L2_weight 0.001  --zd_kl_weight 1e-5  --zc_kl_weight 1e-5 --sparsity_weight 1e-5 --pred_len 48 '
            ' --depth 9 --hidden_dim 256 --hidden_layers 1 --tau 0.7 --batch_size 8 --learning_rate 0.001 --mode var --x_dim 10 '
    },

    'ECL': {
        1: ' --method LSTD_TOT --root_path ./data/ --data ECL --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
           ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
           ' --L2_weight 0.001  --zd_kl_weight 1e-5  --zc_kl_weight 1e-5 --sparsity_weight 1e-5 --pred_len 1 '
           ' --depth 9 --hidden_dim 512 --hidden_layers 1 --tau 0.75 --batch_size 4 --learning_rate 0.002 --mode time --x_dim 13 ',
        24: ' --method LSTD_TOT --root_path ./data/ --data ECL --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
            ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
            ' --L2_weight 0.001  --zd_kl_weight 1e-5  --zc_kl_weight 1e-5 --sparsity_weight 1e-5 --pred_len 24 '
            ' --depth 9 --hidden_dim 512 --hidden_layers 1 --tau 0.75 --batch_size 4 --learning_rate 0.002 --mode var --x_dim 17 ',
        48: ' --method LSTD_TOT --root_path ./data/ --data ECL --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 17 '
            ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
            ' --L2_weight 0.001  --zd_kl_weight 1e-5  --zc_kl_weight 1e-5 --sparsity_weight 1e-5 --pred_len 48 '
            ' --depth 9 --hidden_dim 512 --hidden_layers 1 --tau 0.75 --batch_size 4 --learning_rate 0.002 --mode var --x_dim 18  '
    },

    'ETTh2': {
        1: ' --method LSTD_TOT --root_path ./data/ --data ETTh2 --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
           ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
           ' --L2_weight 0.001  --zd_kl_weight 1e-5  --zc_kl_weight 1e-5 --sparsity_weight 1e-5 --pred_len 1 '
           ' --depth 9 --hidden_dim 512 --hidden_layers 1 --tau 0.75 --batch_size 4 --learning_rate 0.003 --mode time --x_dim 3 ',
        24: ' --method LSTD_TOT --root_path ./data/ --data ETTh2 --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
            ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
            ' --L2_weight 0.001  --zd_kl_weight 1e-5  --zc_kl_weight 1e-5 --sparsity_weight 1e-5 --pred_len 24 '
            ' --depth 9 --hidden_dim 500 --hidden_layers 1 --tau 0.75 --batch_size 4 --learning_rate 0.003 --mode var --x_dim 13',
        48: ' --method LSTD_TOT --root_path ./data/ --data ETTh2 --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
            ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
            ' --L2_weight 0.001  --zd_kl_weight 1e-5  --zc_kl_weight 1e-5 --sparsity_weight 1e-5 --pred_len 48 '
            ' --depth 9 --hidden_dim 490 --hidden_layers 2 --tau 0.7 --batch_size 32 --learning_rate 0.003 --mode var --x_dim 13  '

    },
    'ETTm1': {
        1: ' --method LSTD_TOT --root_path ./data/ --data ETTm1 --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
           ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
           ' --L2_weight 0.001  --zd_kl_weight 1e-5  --zc_kl_weight 1e-5 --sparsity_weight 1e-5 --pred_len 1 '
           ' --depth 9 --hidden_dim 450 --hidden_layers 1 --tau 0.75 --batch_size 8 --learning_rate 0.003 --mode time  --x_dim 13',
        24: ' --method LSTD_TOT --root_path ./data/ --data ETTm1 --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
            ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
            ' --L2_weight 0.001  --zd_kl_weight 1e-5  --zc_kl_weight 1e-5 --sparsity_weight 1e-5 --pred_len 24 '
            ' --depth 9 --hidden_dim 256 --hidden_layers 1 --tau 0.7 --batch_size 4 --learning_rate 0.003 --mode var --x_dim 7 ',
        48: ' --method LSTD_TOT --root_path ./data/ --data ETTm1 --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
            ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
            ' --L2_weight 0.001  --zd_kl_weight 1e-5  --zc_kl_weight 1e-5 --sparsity_weight 1e-5 --pred_len 48 '
            ' --depth 9 --hidden_dim 256 --hidden_layers 1 --tau 0.7 --batch_size 4 --learning_rate 0.002 --mode var --x_dim 13 '
    },

}
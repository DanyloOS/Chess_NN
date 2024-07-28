# Chess_NN
cmd.exe \"/K\" D:\\anaconda\\Scripts\\activate.bat D:\\anaconda

D:\\anaconda\\Scripts\\activate.bat D:\\anaconda
conda activate chess_env
tensorboard --logdir=lightning_logs\\chessml

python nn_test.py model_state_batch_size-1024-layer_count-4-max_epochs-3.bin 4
<!-- cat fen_tests_clean.txt | python nn_test.py model_state_batch_size-1024-layer_count-4-max_epochs-3.bin 4 -->
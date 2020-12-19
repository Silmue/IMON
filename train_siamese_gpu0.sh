nohup python train.py -d datasets/liver.json -b Siamese -g 0 -r 1 --depth 4 --n_pred 3 --loss CC >>gpu0.out 2>&1 &
tail -f gpu0.out
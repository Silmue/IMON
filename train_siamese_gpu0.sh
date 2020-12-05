nohup python train.py -d datasets/liver.json -c weights/Dec03-0933 -b Siamese -g 0 -r 1 --depth 5 --n_pred 3 >>gpu0.out 2>&1 &
tail -f gpu0.out
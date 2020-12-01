nohup python train.py -d datasets/brain.json -b Siamese -g 1 -r 1 --depth 5 --n_pred 3 >>gpu1.out 2>&1 &
tail -f gpu1.out
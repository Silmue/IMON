nohup python train.py -d datasets/brain.json -b VTN -g 0 -r 1 --depth 5 --n_pred 3 >>gpu0.out 2>&1 &
tail -f gpu0.out
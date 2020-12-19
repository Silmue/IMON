nohup python train.py -d datasets/liver.json -b Siamese -g 1 -r 1 --depth 4 --n_pred 3 --loss CC --fake_batch 4 >>gpu1.out 2>&1 &
tail -f gpu1.out
nohup python train.py -d datasets/lpba.json -b VTN -g 1 -r 1 --discriminator FullDiscriminator >>gpu1.out 2>&1 &
tail -f gpu1.out
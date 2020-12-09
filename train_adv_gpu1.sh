nohup python train.py -d datasets/brain.json --prestep 60000 -b VTN --discriminator PartDiscriminator -g 1 -r 1 >>gpu1.out 2>&1 &
tail -f gpu1.out
nohup python train.py -d datasets/brain.json -c weights/Dec08-2224 --clear_steps -b VTN --discriminator PartDiscriminator -g 1 -r 1 >>gpu1.out 2>&1 &
tail -f gpu1.out
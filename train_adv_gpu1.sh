nohup python train.py -d datasets/brain.json -c weights/Dec02-2209 -b VTN -g 1 -r 1 --discriminator PartDiscriminator >>gpu1.out 2>&1 &
tail -f gpu1.out
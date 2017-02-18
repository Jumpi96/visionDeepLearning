# Project - Deep Learning - Udacity Course

## Step 1: Design and test a model architecture that can identify sequences of digits in an image.

After some failures trying to use an Inception module (by Google) or another architecture of ConvNets, I finished working with a classic LeNet architecture.
In next steps, it is going to work as a recurrent network that emit the sequence of digits one-at-a-time.
The model was optimized working with NMIST dataset. Best accuracy achieved on test dataset: 92%.
I've tried to apply dropout but it drops accuracy considerably.

## Step 2: Train a model on a realistic dataset.

I'm working with a similar architecture. I have implemented dropout and learning rate decay to optimize it. I've always created a balanced cross-validation set (see Sources).
Note: I am having some problems to train my ConvNet because of my PC. I'm going to fix it on these days using another one. This dataset is huge (600k images for training).

## Next: sequence of digits and Android implementation.

I will continue working on this on the next weeks.

## Sources:
	[Udacity's Deep Learning Course](https://www.udacity.com/course/deep-learning--ud7309)
	[Thomalm GitHub](https://github.com/thomalm/svhn-multi-digit/)

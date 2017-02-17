Project - Deep Learning - Udacity Course

Step 1: Design and test a model architecture that can identify sequences of digits in an image.

After some failures trying to use an Inception module (by Google) or another architecture of ConvNets, I finished working with a classic LeNet architecture.
In next steps, it is going to work as a recurrent network that emit the sequence of digits one-at-a-time.
The model was optimized working with NMIST dataset. Best accuracy on test dataset: 92%.
I've tried to apply dropout but it drops accuracy considerably.

Step 2: Train a model on a realistic dataset.

Previous model does not work. It is not possible to make it to converge. I have tried changing hyper-parametres but it is not working.

TRY:
	https://github.com/thomalm/svhn-multi-digit/blob/master/01-svhn-single-preprocessing.ipynb
	
	To-Do
	
	
	
	



# PoseNet-Implementation-for-Calib-Challenge

- To solve the calib challenge [PoseNet](https://arxiv.org/pdf/1505.07427.pdf) architecture was implemented with regularized geometric loss functions for pose regression for which motivation was found from this [paper](https://arxiv.org/abs/1704.00390).

- Compared to this implementation a [much smaller neural network](https://github.com/Msarang7/Calib-Challenge-An-Attempt) converged faster and gave only 2% error on training dataset. Hence, this experiment was discarded. It can work better with datasets having more varied frames along with the translational attributes.

- The regularized loss function used in the paper takes into consideration the rotational as well as translational loss while the dataset to work on had only rotational attributes (Euler's angles). The regularizing parameters in the loss function were also learned during the training process.

- Pitch and Yaw angles were provided for training. To convert them to quaternions roll was assumed to be equal to zero. 

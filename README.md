# Loss landscape

Keras implementation of [loss-landscape](https://github.com/tomgoldstein/loss-landscape) and reproduction of results from [this paper](https://arxiv.org/abs/1712.09913).

Using code from this repo you can visualize the loss value around the learned set of parameters. In this case we treat the set of network parameters at one point. We evaluate the loss values around that point and plot it in 2d or 3d. To form a grid of x-y points around learned parameters (or a set of points on the line) we need to specify a direction of x or y axis. Here we use random direction with filter-wise normalization. For a more detailed explanation with mathematical background, please read [the introductory work](https://arxiv.org/abs/1712.09913).

Please be aware, that during the evaluation of loss values, the weights values of your model will be changed. If you don't want to lose your weights, please create checkpoint of your model before calling any of below functions.

## Interface

All functions from public interface are in [losslandscape.py](losslandscape.py) file. The interface is quite simple. It focuses on the type of plots you can obtain. Plots in this section may have a weird shape. They are just a sample generated for interface demonstration, the networks were not trained properly. For more reasonable results, please go to section 'Basic healthchecks' or 'Reproduction of paper results'.

### 2D plots

Simplest 2D plot can be generated using:
```
plot_loss(model, (x_test, y_test))
```
![](readme_files/2d_example1.png)
In this code <code>model</code> is instance of trained neural network. Tuple <code>(x_test, y_test)</code> is the dataset that will be used for loss evaluation. 
  
We can generate plots for both training and test set:
```
plot_loss(model, ((x_train, y_train), (x_test, y_test)), dataset_labels=('training set', 'test set'))
```
![](readme_files/2d_example2.png)
When providing <code>dataset_labels</code> number of datasets and number of labels must be equal. In the other case <code>AssertionError</code> will be raised. Specifying <code>dataset_labels</code> is not necessary.

To avoid long computation or to get more precise plot you can specify the number of points in which loss will be evaluated:
```
plot_loss(model, (x_test, y_test), number_of_points=41)
```
![](readme_files/2d_example3.png)
This will affect the duration of computations and the quality of obtained plot. To low <code>number_of_points</code> may hide some nuances of loss shape. Default value of <code>number_of_points</code> is 21.


### 3D plots

We can generate two types of plots: levels and 3d.
Here is an example of levels plot:
```
plot_loss_3D(model, "levels", x_test, y_test)
```
![](readme_files/3d_example1.png)

It is analogous to 2D case, but the second argument is type of plot. 

Similarly the 3d plots can be obtained with:
```
plot_loss_3D(model, "3d", x_test, y_test)
```
![](readme_files/3d_example2.png)

If you are interest in both levels and 3d you can get them in one call:
```
plot_loss_3D(model, ("levels", "3d"), x_test, y_test)
```
This will result in two separate figures containing specified plots.

As in 2d case there is possibility to specify number of points. Here number of points is defined per axis, so the x-y grid of plot will have <code>number_of_points**2</code> points.
```
plot_loss_3D(model, "levels", x_test, y_test, number_of_points=5)
```
![](readme_files/3d_example3.png)

## Basic healthchecks

Here are plots based sanity checks from [Tom Goldstein](https://github.com/tomgoldstein) [lecture](https://www.youtube.com/watch?v=78vq6kgsTa8). To obtain this plots I used ResNet with 56 layers trained on cifar10 (with standard train-test split). All plots are obtained with the test set.

### Batch size

 Networks with large batch sizes are known to find 'worse' optimum with higher generalization error. In loss landscape plots it correspond to narrower minimum. For smaller batch size we should observe wider plot.

batch size = 128, test set acc = 0.9250
![](readme_files/healthcheck_batchsize_128.png)
batch size = 2048, test set acc = 0.8765
![](readme_files/healthcheck_batchsize_2048.png)

### Weight decay

Properly tuned weight decay should decrease the generalization error (as any regularization technique). Again, this should be visible as wider minimum.

weight decay = 0.0005, test set acc = 0.9292
![](readme_files/healthcheck_weight_decay_0005.png)
no weight decay, test set acc = 0.8946
![](readme_files/healthcheck_weight_decay_None.png)

### Optimizer

SGD optimizer can find minimums with the lower generalization error. This should be visible in 2d plots of loss landscape.

SGD optimizer, test set acc = 0.9283
![](readme_files/healthcheck_optimizer_SGD.png)
Adam optimizer, test set acc = 0.7958
![](readme_files/healthcheck_optimizer_Adam.png)


Of all 3 comparisons only for batch normalization the shape of the loss around learned parameters is not quite as we expected. This may be due to more shallow shape of the minimum for the bigger batch. Also the minimum for lower batch size is more narrow compared to other experiments. Something may be wrong with the values of hyperparameters for this experiment, although obtained values of accuracy are in line with our intuition.

## Reproduction of paper results

ResNets are known to benefit greatly from increased number of layers. In traditional vgg-like networks the excessive depth of layer can cripple the network performance. This property can be observed in the 3d plots of the loss landscape. 

With increased depth of vgg-like network, the loss landscape around minimum should be more chaotic. This corresponds to wider minimum from previous section. We expect the loss landscape around vgg to become less convex as we add more layers. On the other hand, the loss landscape around the ResNet should remain convex.

ResNet20
![](readme_files/resnet20.png)

ResNet56
![](readme_files/resnet56.png)

ResNet110
![](readme_files/resnet110.png)

ResNet20
![](readme_files/resnet20_3d.png)

ResNet56
![](readme_files/resnet56_3d.png)

ResNet110
![](readme_files/resnet110_3d.png)


VGG20
![](readme_files/vgglike20.png)

VGG56
![](readme_files/vgglike56.png)

VGG20
![](readme_files/vgglike20_3d.png)

VGG56
![](readme_files/vgglike56_3d.png)

Plots for vgg with 110 layers were not obtained due to extensive computation power requirements. They will be added in future. Despite lack of the last plot, even for vgg56 we can see more non-convexity than for ResNet56 or ResNet110, and smaller convex-shaped area of minimum in general. This can increase confidence, that in this plots we can observe more valid representation of loss landscape.
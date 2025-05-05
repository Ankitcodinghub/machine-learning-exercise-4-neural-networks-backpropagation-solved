# machine-learning-exercise-4-neural-networks-backpropagation-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning Exercise 4-Neural Networks Backpropagation Solved](https://www.ankitcodinghub.com/product/machine-learning-exercise-4-neural-networks-backpropagation-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;94650&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning Exercise 4-Neural Networks Backpropagation Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
Programming Exercise 4: Neural Networks Learning

Machine Learning

Introduction

In this exercise, you will implement the backpropagation algorithm for neural networks and apply it to the task of hand-written digit recognition. Before starting on the programming exercise, we strongly recommend watching the video lectures and completing the review questions for the associated topics.

To get started with the exercise, you will need to download the starter code and unzip its contents to the directory where you wish to complete the exercise. If needed, use the cd command in Octave/MATLAB to change to this directory before starting this exercise.

You can also find instructions for installing Octave/MATLAB in the “En- vironment Setup Instructions” of the course website.

Files included in this exercise

ex4.m – Octave/MATLAB script that steps you through the exercise ex4data1.mat – Training set of hand-written digits

ex4weights.mat – Neural network parameters for exercise 4 submit.m – Submission script that sends your solutions to our servers displayData.m – Function to help visualize the dataset

fmincg.m – Function minimization routine (similar to fminunc) sigmoid.m – Sigmoid function

computeNumericalGradient.m – Numerically compute gradients checkNNGradients.m – Function to help check your gradients debugInitializeWeights.m – Function for initializing weights predict.m – Neural network prediction function

[⋆] sigmoidGradient.m – Compute the gradient of the sigmoid function [⋆] randInitializeWeights.m – Randomly initialize weights

[⋆] nnCostFunction.m – Neural network cost function

</div>
</div>
<div class="layoutArea">
<div class="column">
1

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
⋆ indicates files you will need to complete

Throughout the exercise, you will be using the script ex4.m. These scripts set up the dataset for the problems and make calls to functions that you will write. You do not need to modify the script. You are only required to modify functions in other files, by following the instructions in this assignment.

Where to get help

The exercises in this course use Octave1 or MATLAB, a high-level program- ming language well-suited for numerical computations. If you do not have Octave or MATLAB installed, please refer to the installation instructions in the “Environment Setup Instructions” of the course website.

At the Octave/MATLAB command line, typing help followed by a func- tion name displays documentation for a built-in function. For example, help plot will bring up help information for plotting. Further documentation for Octave functions can be found at the Octave documentation pages. MAT- LAB documentation can be found at the MATLAB documentation pages.

We also strongly encourage using the online Discussions to discuss ex- ercises with other students. However, do not look at any source code written by others or share your source code with others.

1 Neural Networks

In the previous exercise, you implemented feedforward propagation for neu- ral networks and used it to predict handwritten digits with the weights we provided. In this exercise, you will implement the backpropagation algorithm to learn the parameters for the neural network.

The provided script, ex4.m, will help you step through this exercise. 1.1 Visualizing the data

In the first part of ex4.m, the code will load the data and display it on a 2-dimensional plot (Figure 1) by calling the function displayData.

1Octave is a free alternative to MATLAB. For the programming exercises, you are free to use either Octave or MATLAB.

</div>
</div>
<div class="layoutArea">
<div class="column">
2

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
Figure 1: Examples from the dataset

This is the same dataset that you used in the previous exercise. There are 5000 training examples in ex3data1.mat, where each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity at that location. The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector. Each of these training examples becomes a single row in our data matrix X. This gives us a 5000 by 400 matrix X where every row is a training example for a handwritten digit image.

 —(x(1))T —   —(x(2))T — 

X =  .  .

— (x(m))T —

The second part of the training set is a 5000-dimensional vector y that contains labels for the training set. To make things more compatible with Octave/MATLAB indexing, where there is no zero index, we have mapped the digit zero to the value ten. Therefore, a “0” digit is labeled as “10”, while the digits “1” to “9” are labeled as “1” to “9” in their natural order.

1.2 Model representation

Our neural network is shown in Figure 2. It has 3 layers – an input layer, a hidden layer and an output layer. Recall that our inputs are pixel values

</div>
</div>
<div class="layoutArea">
<div class="column">
3

</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
of digit images. Since the images are of size 20 × 20, this gives us 400 input layer units (not counting the extra bias unit which always outputs +1). The training data will be loaded into the variables X and y by the ex4.m script.

You have been provided with a set of network parameters (Θ(1),Θ(2)) already trained by us. These are stored in ex4weights.mat and will be loaded by ex4.m into Theta1 and Theta2. The parameters have dimensions that are sized for a neural network with 25 units in the second layer and 10 output units (corresponding to the 10 digit classes).

<pre>% Load saved matrices from file
</pre>
<pre>load('ex4weights.mat');
</pre>
<pre>% The matrices Theta1 and Theta2 will now be in your workspace
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
</pre>
Figure 2: Neural network model.

1.3 Feedforward and cost function

Now you will implement the cost function and gradient for the neural net- work. First, complete the code in nnCostFunction.m to return the cost.

</div>
</div>
<div class="layoutArea">
<div class="column">
4

</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="layoutArea">
<div class="column">
Recall that the cost function for the neural network (without regulariza- tion) is

</div>
</div>
<div class="layoutArea">
<div class="column">
1mK􏰌 􏰍 J(θ)= 􏰉􏰉 −y(i)log((hθ(x(i)))k)−(1−y(i))log(1−(hθ(x(i)))k) ,

</div>
</div>
<div class="layoutArea">
<div class="column">
mkk i=1 k=1

</div>
</div>
<div class="layoutArea">
<div class="column">
where hθ(x(i)) is computed as shown in the Figure 2 and K = 10 is the total number of possible labels. Note that hθ(x(i))k = a(3) is the activation (output

value) of the k-th output unit. Also, recall that whereas the original labels (in the variable y) were 1, 2, …, 10, for the purpose of training a neural network, we need to recode the labels as vectors containing only values 0 or 1, so that

</div>
</div>
<div class="layoutArea">
<div class="column">
1 0

0 y= ,

 .  

</div>
</div>
<div class="layoutArea">
<div class="column">
0 1

0

 , … or

</div>
</div>
<div class="layoutArea">
<div class="column">
0 0

0  .

For example, if x(i) is an image of the digit 5, then the corresponding y(i) (that you should use with the cost function) should be a 10-dimensional vector with y5 = 1, and the other elements equal to 0.

You should implement the feedforward computation that computes hθ(x(i)) for every example i and sum the cost over all examples. Your code should also work for a dataset of any size, with any number of labels (you can assume that there are always at least K ≥ 3 labels).

</div>
</div>
<div class="layoutArea">
<div class="column">
 .  001

</div>
</div>
<div class="layoutArea">
<div class="column">
Once you are done, ex4.m will call your nnCostFunction using the loaded set of parameters for Theta1 and Theta2. You should see that the cost is about 0.287629.

</div>
</div>
<div class="layoutArea">
<div class="column">
5

</div>
</div>
<div class="layoutArea">
<div class="column">
k

</div>
</div>
<div class="layoutArea">
<div class="column">
 .  

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
Implementation Note: The matrix X contains the examples in rows (i.e., X(i,:)’ is the i-th training example x(i), expressed as a n × 1 vector.) When you complete the code in nnCostFunction.m, you will need to add the column of 1’s to the X matrix. The parameters for each unit in the neural network is represented in Theta1 and Theta2 as one row. Specifically, the first row of Theta1 corresponds to the first hidden unit in the second layer. You can use a for-loop over the examples to compute the cost.

</div>
</div>
</div>
</div>
<div class="page" title="Page 6">
<div class="layoutArea">
<div class="column">
You should now submit your solutions.

1.4 Regularized cost function

The cost function for neural networks with regularization is given by

</div>
</div>
<div class="layoutArea">
<div class="column">
J(θ) =

</div>
<div class="column">
1mK􏰌 􏰍 􏰉 􏰉 −y(i) log((hθ(x(i)))k) − (1 − y(i)) log(1 − (hθ(x(i)))k) +

</div>
</div>
<div class="layoutArea">
<div class="column">
mkk i=1 k=1

</div>
</div>
<div class="layoutArea">
<div class="column">
λ􏰊25 400 10 25 􏰋 􏰉􏰉(Θ(1))2 +􏰉􏰉(Θ(2))2

2m j,k j,k j=1 k=1 j=1 k=1

</div>
<div class="column">
.

</div>
</div>
<div class="layoutArea">
<div class="column">
You can assume that the neural network will only have 3 layers – an input layer, a hidden layer and an output layer. However, your code should work for any number of input units, hidden units and outputs units. While we have explicitly listed the indices above for Θ(1) and Θ(2) for clarity, do note that your code should in general work with Θ(1) and Θ(2) of any size.

Note that you should not be regularizing the terms that correspond to the bias. For the matrices Theta1 and Theta2, this corresponds to the first column of each matrix. You should now add regularization to your cost function. Notice that you can first compute the unregularized cost function J using your existing nnCostFunction.m and then later add the cost for the regularization terms.

Once you are done, ex4.m will call your nnCostFunction using the loaded set of parameters for Theta1 and Theta2, and λ = 1. You should see that the cost is about 0.383770.

You should now submit your solutions.

2 Backpropagation

In this part of the exercise, you will implement the backpropagation algo- rithm to compute the gradient for the neural network cost function. You will need to complete the nnCostFunction.m so that it returns an appropri- ate value for grad. Once you have computed the gradient, you will be able to train the neural network by minimizing the cost function J(Θ) using an advanced optimizer such as fmincg.

You will first implement the backpropagation algorithm to compute the gradients for the parameters for the (unregularized) neural network. After

</div>
</div>
<div class="layoutArea">
<div class="column">
6

</div>
</div>
</div>
<div class="page" title="Page 7">
<div class="layoutArea">
<div class="column">
you have verified that your gradient computation for the unregularized case is correct, you will implement the gradient for the regularized neural network.

2.1 Sigmoid gradient

To help you get started with this part of the exercise, you will first implement the sigmoid gradient function. The gradient for the sigmoid function can be computed as

</div>
</div>
<div class="layoutArea">
<div class="column">
where

</div>
<div class="column">
g′(z)= dg(z)=g(z)(1−g(z)) dz

sigmoid(z) = g(z) = 1 . 1+e−z

</div>
</div>
<div class="layoutArea">
<div class="column">
When you are done, try testing a few values by calling sigmoidGradient(z) at the Octave/MATLAB command line. For large values (both positive and negative) of z, the gradient should be close to 0. When z = 0, the gradi- ent should be exactly 0.25. Your code should also work with vectors and matrices. For a matrix, your function should perform the sigmoid gradient function on every element.

You should now submit your solutions.

2.2 Random initialization

When training neural networks, it is important to randomly initialize the pa- rameters for symmetry breaking. One effective strategy for random initializa- tion is to randomly select values for Θ(l) uniformly in the range [−εinit, εinit]. You should use εinit = 0.12.2 This range of values ensures that the parameters are kept small and makes the learning more efficient.

Your job is to complete randInitializeWeights.m to initialize the weights for Θ; modify the file and fill in the following code:

<pre>% Randomly initialize the weights to small values
</pre>
epsilon init = 0.12;

W = rand(L out, 1 + L in) * 2 * epsilon init − epsilon init;

You do not need to submit any code for this part of the exercise.

2One effective strategy for choosing εinit is to base it on the number of units in the √

network. A good choice of εinit is εinit = √ 6 , where Lin = sl and Lout = sl+1 are Lin +Lout

the number of units in the layers adjacent to Θ(l). 7

</div>
</div>
</div>
<div class="page" title="Page 8">
<div class="layoutArea">
<div class="column">
2.3 Backpropagation

Figure 3: Backpropagation Updates.

Now, you will implement the backpropagation algorithm. Recall that

the intuition behind the backpropagation algorithm is as follows. Given a

training example (x(t),y(t)), we will first run a “forward pass” to compute

all the activations throughout the network, including the output value of the

hypothesis hΘ(x). Then, for each node j in layer l, we would like to compute

an “error term” δ(l) that measures how much that node was “responsible” j

for any errors in our output.

For an output node, we can directly measure the difference between the

network’s activation and the true target value, and use that to define δ(3) j

(since layer 3 is the output layer). For the hidden units, you will compute δ(l) based on a weighted average of the error terms of the nodes in layer

(l + 1).

In detail, here is the backpropagation algorithm (also depicted in Figure

3). You should implement steps 1 to 4 in a loop that processes one example at a time. Concretely, you should implement a for-loop for t = 1:m and place steps 1-4 below inside the for-loop, with the tth iteration performing the calculation on the tth training example (x(t),y(t)). Step 5 will divide the accumulated gradients by m to obtain the gradients for the neural network cost function.

</div>
</div>
<div class="layoutArea">
<div class="column">
j

</div>
</div>
<div class="layoutArea">
<div class="column">
8

</div>
</div>
</div>
<div class="page" title="Page 9">
<div class="layoutArea">
<div class="column">
1. Set the input layer’s values (a(1)) to the t-th training example x(t).

Perform a feedforward pass (Figure 2), computing the activations (z(2), a(2), z(3), a(3)) for layers 2 and 3. Note that you need to add a +1 term to ensure that

the vectors of activations for layers a(1) and a(2) also include the bias

unit. In Octave/MATLAB, if a 1 is a column vector, adding one corre- spondstoa1 = [1 ; a1].

2. For each output unit k in layer 3 (the output layer), set δ(3) = (a(3) − yk),

</div>
</div>
<div class="layoutArea">
<div class="column">
kk

</div>
</div>
<div class="layoutArea">
<div class="column">
where yk ∈ {0,1} indicates whether the current training example be- longs to class k (yk = 1), or if it belongs to a different class (yk = 0). You may find logical arrays helpful for this task (explained in the pre- vious programming exercise).

3. For the hidden layer l = 2, set

δ(2) = 􏰀Θ(2)􏰁T δ(3). ∗ g′(z(2))

4. Accumulate the gradient from this example using the following for- mula. Note that you should skip or remove δ(2). In Octave/MATLAB,

removing δ(2) corresponds to delta 2 = delta 2(2:end). 0

∆(l) = ∆(l) + δ(l+1)(a(l))T

5. Obtain the (unregularized) gradient for the neural network cost func-

tion by dividing the accumulated gradients by m1 :

</div>
</div>
<div class="layoutArea">
<div class="column">
∂ J(Θ)=D(l) = 1∆(l)

</div>
</div>
<div class="layoutArea">
<div class="column">
0

</div>
</div>
<div class="layoutArea">
<div class="column">
∂Θ(l) ij

</div>
<div class="column">
ij m ij

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
Octave/MATLAB Tip: You should implement the backpropagation algorithm only after you have successfully completed the feedforward and cost functions. While implementing the backpropagation algorithm, it is often useful to use the size function to print out the sizes of the vari- ables you are working with if you run into dimension mismatch errors (“nonconformant arguments” errors in Octave/MATLAB).

</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
9

</div>
</div>
</div>
<div class="page" title="Page 10">
<div class="layoutArea">
<div class="column">
After you have implemented the backpropagation algorithm, the script ex4.m will proceed to run gradient checking on your implementation. The gradient check will allow you to increase your confidence that your code is computing the gradients correctly.

2.4 Gradient checking

In your neural network, you are minimizing the cost function J(Θ). To perform gradient checking on your parameters, you can imagine “unrolling” the parameters Θ(1), Θ(2) into a long vector θ. By doing so, you can think of the cost function being J(θ) instead and use the following gradient checking procedure.

Suppose you have a function fi(θ) that purportedly computes ∂ J(θ);

</div>
</div>
<div class="layoutArea">
<div class="column">
you’d like to check if fi is outputting correct derivative values.

</div>
<div class="column">
∂θi

</div>
</div>
<div class="layoutArea">
<div class="column">
(i+)  . 

</div>
<div class="column">
(i−)

and θ =θ− 

</div>
</div>
<div class="layoutArea">
<div class="column">
0  0. 

</div>
<div class="column">
0  0. 

</div>
</div>
<div class="layoutArea">
<div class="column">
.

</div>
<div class="column">
.  . 

</div>
</div>
<div class="layoutArea">
<div class="column">
Let θ =θ+ ε.

 . . 00

</div>
</div>
<div class="layoutArea">
<div class="column">
 

</div>
<div class="column">
ε.   . . 

</div>
</div>
<div class="layoutArea">
<div class="column">


</div>
</div>
<div class="layoutArea">
<div class="column">
So, θ(i+) is the same as θ, except its i-th element has been incremented by ε. Similarly, θ(i−) is the corresponding vector with the i-th element decreased by ε. You can now numerically verify fi(θ)’s correctness by checking, for each i, that:

fi(θ) ≈ J(θ(i+)) − J(θ(i−)). 2ε

The degree to which these two values should approximate each other will depend on the details of J. But assuming ε = 10−4, you’ll usually find that the left- and right-hand sides of the above will agree to at least 4 significant digits (and often many more).

We have implemented the function to compute the numerical gradient for you in computeNumericalGradient.m. While you are not required to modify the file, we highly encourage you to take a look at the code to understand how it works.

In the next step of ex4.m, it will run the provided function checkNNGradients.m which will create a small neural network and dataset that will be used for checking your gradients. If your backpropagation implementation is correct,

</div>
</div>
<div class="layoutArea">
<div class="column">
10

</div>
</div>
</div>
<div class="page" title="Page 11">
<div class="layoutArea">
<div class="column">
you should see a relative difference that is less than 1e-9.

</div>
</div>
<table>
<tbody>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Practical Tip: When performing gradient checking, it is much more efficient to use a small neural network with a relatively small number of input units and hidden units, thus having a relatively small number of parameters. Each dimension of θ requires two evaluations of the cost function and this can be expensive. In the function checkNNGradients, our code creates a small random model and dataset which is used with computeNumericalGradient for gradient checking. Furthermore, after you are confident that your gradient computations are correct, you should turn off gradient checking before running your learning algorithm.

</div>
</div>
</td>
</tr>
<tr>
<td>
<div class="layoutArea">
<div class="column">
Practical Tip: Gradient checking works for any function where you are computing the cost and the gradient. Concretely, you can use the same computeNumericalGradient.m function to check if your gradient imple- mentations for the other exercises are correct too (e.g., logistic regression’s cost function).

</div>
</div>
</td>
</tr>
</tbody>
</table>
<div class="layoutArea">
<div class="column">
Once your cost function passes the gradient check for the (unregularized) neural network cost function, you should submit the neural network gradient function (backpropagation).

2.5 Regularized Neural Networks

After you have successfully implemeted the backpropagation algorithm, you will add regularization to the gradient. To account for regularization, it turns out that you can add this as an additional term after computing the gradients using backpropagation.

Specifically, after you have computed ∆(l) using backpropagation, you ij

forj=0 forj≥1

</div>
</div>
<div class="layoutArea">
<div class="column">
should add regularization using

∂ J(Θ)=D(l) = 1∆(l)

</div>
</div>
<div class="layoutArea">
<div class="column">
∂Θ(l) ij m ij ij

</div>
</div>
<div class="layoutArea">
<div class="column">
∂ J(Θ)=D(l)=1∆(l)+λΘ(l) ∂Θ(l) ij m ij m ij

</div>
</div>
<div class="layoutArea">
<div class="column">
ij

Note that you should not be regularizing the first column of Θ(l) which

is used for the bias term. Furthermore, in the parameters Θ(l), i is indexed ij

</div>
</div>
<div class="layoutArea">
<div class="column">
11

</div>
</div>
</div>
<div class="page" title="Page 12">
<div class="layoutArea">
<div class="column">
starting from 1, and j is indexed starting from 0. Thus,

</div>
</div>
<div class="layoutArea">
<div class="column">
Θ(l) =Θ(i) 2,0

</div>
</div>
<div class="layoutArea">
<div class="column">
Θ(i) 1,0

</div>
<div class="column">
Θ(l) 1,1 Θ(l) 2,1

</div>
<div class="column">
. . .

</div>
</div>
<div class="layoutArea">
<div class="column">
.

Somewhat confusingly, indexing in Octave/MATLAB starts from 1 (for

</div>
</div>
<div class="layoutArea">
<div class="column">
.

both i and j), thus Theta1(2, 1) actually corresponds to Θ(l) (i.e., the entry

</div>
</div>
<div class="layoutArea">
<div class="column">
…

Now modify your code that computes grad in nnCostFunction to account for regularization. After you are done, the ex4.m script will proceed to run gradient checking on your implementation. If your code is correct, you should expect to see a relative difference that is less than 1e-9.

You should now submit your solutions.

2.6 Learning parameters using fmincg

After you have successfully implemented the neural network cost function and gradient computation, the next step of the ex4.m script will use fmincg to learn a good set parameters.

After the training completes, the ex4.m script will proceed to report the training accuracy of your classifier by computing the percentage of examples it got correct. If your implementation is correct, you should see a reported training accuracy of about 95.3% (this may vary by about 1% due to the random initialization). It is possible to get higher training accuracies by training the neural network for more iterations. We encourage you to try training the neural network for more iterations (e.g., set MaxIter to 400) and also vary the regularization parameter λ. With the right learning settings, it is possible to get the neural network to perfectly fit the training set.

3 Visualizing the hidden layer

One way to understand what your neural network is learning is to visualize

what the representations captured by the hidden units. Informally, given a

particular hidden unit, one way to visualize what it computes is to find an

input x that will cause it to activate (that is, to have an activation value

(a(l)) close to 1). For the neural network you trained, notice that the ith row i

of Θ(1) is a 401-dimensional vector that represents the parameter for the ith 12

</div>
</div>
<div class="layoutArea">
<div class="column">
2,0

in the second row, first column of the matrix Θ(1) shown above)

</div>
</div>
</div>
<div class="page" title="Page 13">
<div class="layoutArea">
<div class="column">
hidden unit. If we discard the bias term, we get a 400 dimensional vector that represents the weights from each input pixel to the hidden unit.

Thus, one way to visualize the “representation” captured by the hidden unit is to reshape this 400 dimensional vector into a 20 × 20 image and display it.3 The next step of ex4.m does this by using the displayData function and it will show you an image (similar to Figure 4) with 25 units, each corresponding to one hidden unit in the network.

In your trained network, you should find that the hidden units corre- sponds roughly to detectors that look for strokes and other patterns in the input.

Figure 4: Visualization of Hidden Units.

3.1 Optional (ungraded) exercise

In this part of the exercise, you will get to try out different learning settings for the neural network to see how the performance of the neural network varies with the regularization parameter λ and number of training steps (the MaxIter option when using fmincg).

Neural networks are very powerful models that can form highly complex decision boundaries. Without regularization, it is possible for a neural net- work to “overfit” a training set so that it obtains close to 100% accuracy on the training set but does not as well on new examples that it has not seen before. You can set the regularization λ to a smaller value and the MaxIter parameter to a higher number of iterations to see this for youself.

3It turns out that this is equivalent to finding the input that gives the highest activation for the hidden unit, given a “norm” constraint on the input (i.e., ∥x∥2 ≤ 1).

</div>
</div>
<div class="layoutArea">
<div class="column">
13

</div>
</div>
</div>
<div class="page" title="Page 14">
<div class="layoutArea">
<div class="column">
You will also be able to see for yourself the changes in the visualizations of the hidden units when you change the learning parameters λ and MaxIter.

You do not need to submit any solutions for this optional (ungraded) exercise.

</div>
</div>
<div class="layoutArea">
<div class="column">
14

</div>
</div>
</div>
<div class="page" title="Page 15">
<div class="layoutArea">
<div class="column">
Submission and Grading

After completing various parts of the assignment, be sure to use the submit function system to submit your solutions to our servers. The following is a breakdown of how each part of this exercise is scored.

</div>
</div>
<div class="layoutArea">
<div class="column">
Part

Feedforward and Cost Function Regularized Cost Function Sigmoid Gradient

Neural Net Gradient (Backpropagation)

Regularized Gradient Total Points

</div>
<div class="column">
Submitted File

<pre>nnCostFunction.m
nnCostFunction.m
sigmoidGradient.m
nnCostFunction.m
</pre>
<pre>nnCostFunction.m
</pre>
</div>
<div class="column">
Points

30 points 15 points 5 points 40 points

10 points 100 points

</div>
</div>
<div class="layoutArea">
<div class="column">
Function

</div>
</div>
<div class="layoutArea">
<div class="column">
You are allowed to submit your solutions multiple times, and we will take only the highest score into consideration.

</div>
</div>
<div class="layoutArea">
<div class="column">
15

</div>
</div>
</div>

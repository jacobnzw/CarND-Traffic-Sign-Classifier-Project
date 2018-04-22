# **Traffic Sign Recognition** 

---

## The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[hist_norm]: ./examples/hist_normalized.jpg "Histogram of normalized data"
[hist_unnorm]: ./examples/hist_unnormalized.jpg "Histogram of un-normalized data"
[hist_data]: ./examples/hist_data_split.jpg "Class distribution among train, validation and test set."
[testimg_1]: ./webimg/testimg_1.jpg 
[testimg_2]: ./webimg/testimg_2.jpg
[testimg_3]: ./webimg/testimg_3.jpg
[testimg_4]: ./webimg/testimg_4.jpg
[testimg_5]: ./webimg/testimg_5.jpg

---

## Data Set Summary & Exploration

I used the NumPy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images
* The size of the validation set is 12630 images
* The size of test set is 4110 images
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a normalized histogram chart, which gives insight into the class distribution for train set, validation set and test set. 

![][hist_data]

There is a noticable over-representation of the classes with IDs 20, 21 and 22 in the test set. Class distribution in training and validation sets is more or less comparable.

## Model Architecture Design and Testing

### Data Pre-Processing

Right from the onset, I decided to apply as little pre-processing as possible, because it simplifies the whole pipeline and I thought I would rather work getting the right architecture first. So the only pre-processing step that I applied was the data normalization. I tried three different normalization methods, including Tensoflow's own `tf.image.per_image_standardization()`. I ended up choosing my own method, which computes mean and standard deviation of each channel in the image, subtracts the mean and divides by the standard deviation.

Here is comparison of image data histograms before and after normalization.

![alt text][hist_unnorm]

![alt text][hist_norm]

I wanted to keep everything simple from the start and add complexity later. For these reasons I avoided data augmentation. However, as it became apparent later this technique might be indeed very usefull for improving the performance on the test set.


### Model Architecture

I decided to come up with my own architecture, which would draw certain design conventions from other well-known architectures.

At first, I though I'll have 3 kinds of filter sizes, from 7x7 in the first layer down to the 3x3 in the last layer of the feature extraction phase. This architecture however gave poor performance. As I realized later, this was probably due to the fact that I had only one conv-layer for each filter size.

On my second attempt, I dropped the 7x7 conv-layer and instead used two consecutive 5x5 conv-layers terminated with 2x2 maxpool followed by another two 3x3 conv-layers terminated with 2x2 maxpool. I got increase in performance but not near enough to be acceptable. I thought next I should try and tune the classifier part of the neural net (the one with fully-connected layers). I kept fiddling with different numbers of neurons in each layer, but no matter what I did the performance increase was negligible and I couldn't get the validation accuracy above 90%. This made me think that the classifier is flexible enough, it's the representations that need improving. 

I decided to improve the feature extractor by adding one 5x5 conv-layer and one 3x3 conv-layer. Since the model at this point was starting to get a bit deep, I decided to use the batch normalization to help with learning, which I placed before each RELU activation. BN allowed me to use higher learning rates (which is also mentioned in the original article proposing BN). At this point, I was starting to get validation accuracy around 90% after just one epoch. Finally, I added dropout layer after each activation of the fully-connected layers to add some light regularization.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x4 	|
| Batch Normalization	|												|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x8 	|
| Batch Normalization	|												|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x16 	|
| Batch Normalization	|												|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 16x16x16  				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs	16x16x32    |
| Batch Normalization	|												|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x64	|
| Batch Normalization	|												|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, same padding, outputs	16x16x128	|
| Batch Normalization	|												|
| RELU					|												|
| Max pooling			| 2x2 stride, outputs 8x8x128					|
| Flatten				| output 8192									|
| Fully connected		| output 1024  									|
| Batch Normalization	|												|
| RELU					|												|
| Dropout           	| keep probability 0.9					    	|
| Fully connected		| output 512  									|
| Batch Normalization	|												|
| RELU					|												|
| Dropout           	| keep probability 0.9					    	|
| Fully connected		| output 43    									|
| Softmax				|           									|
 


### Training, Validation and Testing

To train the model, I used an Adam optimizer with the recommended heuristic values for the parameters $ \beta_1=0.9 $ and $ \beta_2 = 0.999 $. I found that after adding the batch normalization I could increase the learning rate from 0.001 to 0.01, which eventually worked best. I used batch size of 128 datapoints and trained for 30 epochs.

For the most part, the only parameters I tunned were the learning rate and the batch size. All the Adam hyperparameters were untouched. 

My final model results were:
* training accuracy of 100%
* validation accuracy of 97%
* testing accuracy of 27.9%

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![][testimg_1] ![][testimg_2] ![][testimg_3] ![][testimg_4] ![][testimg_5]

The first image might be difficult to classify because it is a perpective transformation of the 'bumpy road' traffic sign. The perspective transfromation is harder to simulate in data augmentation then the ordinary rigid-body transformations.

Here are the results of the prediction:

| Image			              |     Prediction	        					| 
|:---------------------------:|:-------------------------------------------:| 
| Bumpy road      		      | Bumpy road   								| 
| Road narrows on the right   | Yield										|
| Speed limit (50km/h)	      | No passing for vehicles over 3.5 metric tons|
| Speed limit (30km/h)	      | Speed limit (30km/h)         				|
| Double curve			      | Right-of-way at the next intersection		|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares favorably to the accuracy on the test set of 27.9%. Surprisingly the 'bumby road' sign was classified correctly.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is virtually certain that this is a bumpy road sign (probability of 1.0), and the image does contain a bumpy road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Bumpy road   									| 
| 0.00     				| Wild animals crossing							|
| 0.00					| Dangerous curve to the right		    		|
| 0.00	      			| Stop					 				        |
| 0.00				    | No entry      						    	|


For the second image, 'Road narrows on the right' sign the model predicts

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.59         			| Yield   									    | 
| 0.41     				| Ahead only						        	|
| 0.00					| Road narrows on the right		        		|
| 0.00	      			| Speed limit (100km/h)					 		|
| 0.00				    | Speed limit (80km/h)          		    	|

The model's third most probable hypothesis is correct, however the probability it assigns to it is negligible.

For the third image, 'Speed limit (50km/h)' sign the model predicts

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No passing for vehicles over 3.5 metric tons  | 
| 0.00     				| Roundabout mandatory							|
| 0.00					| Stop		    		                        |
| 0.00	      			| Speed limit (50km/h)					 		|
| 0.00				    | Keep right      						    	|


For the fourth image, 'Speed limit (30km/h)' sign the model predicts

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed limit (30km/h)                          |
| 0.00                  | Yield                                         |
| 0.00                  | Speed limit (80km/h)                          |
| 0.00                  | Roundabout mandatory                          |
| 0.00                  | No vehicles    	    				    	|

The predicts correctly and as in the previous case, the prediction is 100% certain.

For the final image, 'Double curve' sign the model predicts

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.90                  | Right-of-way at the next intersection         |
| 0.10                  | Speed limit (60km/h)                          |
| 0.00                  | Children crossing                             | 
| 0.00                  | Beware of ice/snow                            |
| 0.00                  | Turn left ahead                               |

The correct prediction is not in even among the top 5 model's hypotheses.
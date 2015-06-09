# Description
Implements logistic regression to classify a given set of documents as one of two classes: 0 or 1. 

The loss function used is the logistic loss, from which we derive an update rule that is implemented through stochastic gradient descent. The early stop method is used to detect convergence and prevent overfitting. The system also uses k-fold cross validation for evaluation and will output the average recall, precision and f1 score to terminal.

Optionally, l2-regularization can be added to the calculations by setting the parameter `reg_const` to a non-zero value in `logistic_regression.py`. Currently, the value is set to 0 as initial experiments have shown that it doesn't improve performance on the given dataset for this particular homework assignment.

## Preprocessing
The system uses a bag-of-word approach to classify the documents. Each document is represented as a bag-of-word vector, but since this will generally produce a very sparse vector, only the words (features) with non-zero values are recorded in the feature files. 

Instead of using a simple term frequency approach, inspiration is drawn from the field of information retrieval and each term's [tf-idf](http://en.wikipedia.org/wiki/Tf%E2%80%93idf) value is used instead.

Also, instead of a simple word tokenization performed through `text.split()`, the [Penn Treebank](http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.treebank) approach is used.

# How to run
Run the following commands in a terminal (note that the package `nltk` is required for preprocessing. However, if you only want to run the logistic regression, `main.py` can be run since the generated data from preprocessing is already included in this package):

	python preprocessing.py
	python main.py

If you want to process other documents than the ones provided in this repository, you can specify that through the command line interface. See the descriptions of `preprocessing.py` and `main.py` for further information.

## Data description

### Directory structure
The given documents should be partitioned into subsets and labeled. The document directory structure should be:

    data_root_dir
    |-- s1
        |-- class0
            |-- file101
            |-- file102
            |-- ...
        |-- class1
            |-- file111
            |-- file112
            |-- ...
    |-- s2
        |-- class0
            |-- file201
            |-- file202
            |-- ...
        |-- class1
            |-- file211
            |-- file212
            |-- ...
    |-- ...

Where `s1`, `s2`, etc. are subsets and the subdirectories `class0` and `class1` contain the documents for class 0 and class 1 respectively.

### File (document) format

* line 1: From (email address)
* line 2: Subject
* rest: Body

Example:

	From: admiral@jhunix.hcf.jhu.edu (Steve C Liu)
	Subject: Re: Bring on the O's

	I heard that Eli is selling the team to a group in Cinninati. This would
	help so that the O's could make some real free agent signings in the 
	offseason.
	...

# Requirements

* nltk
	* only required for preprocessing (Penn Treebank word tokenization)
* numpy
* Python 2.7.3

## nltk
After download, the 'punkt' and 'stopwords' modules are required. To download, open a python shell type

	>>> import nltk
	>>> nltk.download()

Then an installation window appears. Go to the 'Models' tab and select 'punkt' from under the 'Identifier' column. Then click Download and it will install the necessary files. Go to the 'Corpora' tab and download 'stopwords'.

# Results
This is the final output of some test runs using different parameter values:

	learning_rate = 20
	reg_const = 0

	========================================
	Averages of 5-fold cross validation
	Recall:    0.968844221106
	Precision: 0.96434999261
	F1 Score:  0.966456061963
	========================================

	real	0m38.678s
	user	0m38.432s
	sys	0m0.096s

.

	learning_rate = 1
	reg_const = 0

	========================================
	Averages of 5-fold cross validation
	Recall:    0.95175879397
	Precision: 0.956805740703
	F1 Score:  0.95416176996
	========================================

	real	1m3.228s
	user	1m2.958s
	sys	0m0.080s

## Example output

This is the terminal output of running `main.py` on Ubuntu 12.04. The `time` command is used to measure the time it takes to run the program.

	$ time python main.py 
	loading feature file ./data/s2_features
	loading feature file ./data/s3_features
	loading feature file ./data/s4_features
	loading feature file ./data/s5_features
	Calculating the loss function value on the validation set ...
	Epoch 0: average loss function value on validation set: 0.362953
	Convergence critera: when difference > -0.005000
	Epoch 10: average loss function value: 0.275196  |  Difference: -0.087758
	Epoch 20: average loss function value: 0.269903  |  Difference: -0.005293
	Epoch 30: average loss function value: 0.267718  |  Difference: -0.002185
	loading feature file ./data/s1_features

	----------------------------------------
	test set         s1_features
	training sets: ['s2_features', 's3_features', 's4_features']
	validation set:  s5_features

	Confusion matrix
	[[ 187.   10.]
	 [  12.  189.]]
	Recall:    0.939698492462
	Precision: 0.94923857868
	F1 Score:  0.944444444444
	----------------------------------------

	loading feature file ./data/s3_features
	loading feature file ./data/s4_features
	loading feature file ./data/s5_features
	loading feature file ./data/s1_features
	Calculating the loss function value on the validation set ...
	Epoch 0: average loss function value on validation set: 0.361451
	Convergence critera: when difference > -0.005000
	Epoch 10: average loss function value: 0.289042  |  Difference: -0.072409
	Epoch 20: average loss function value: 0.282983  |  Difference: -0.006059
	Epoch 30: average loss function value: 0.280237  |  Difference: -0.002745
	loading feature file ./data/s2_features

	----------------------------------------
	test set         s2_features
	training sets: ['s3_features', 's4_features', 's5_features']
	validation set:  s1_features

	Confusion matrix
	[[ 196.    3.]
	 [   3.  196.]]
	Recall:    0.984924623116
	Precision: 0.984924623116
	F1 Score:  0.984924623116
	----------------------------------------

	loading feature file ./data/s1_features
	loading feature file ./data/s4_features
	loading feature file ./data/s5_features
	loading feature file ./data/s2_features
	Calculating the loss function value on the validation set ...
	Epoch 0: average loss function value on validation set: 0.363887
	Convergence critera: when difference > -0.005000
	Epoch 10: average loss function value: 0.274954  |  Difference: -0.088933
	Epoch 20: average loss function value: 0.269577  |  Difference: -0.005377
	Epoch 30: average loss function value: 0.267367  |  Difference: -0.002209
	loading feature file ./data/s3_features

	----------------------------------------
	test set         s3_features
	training sets: ['s1_features', 's4_features', 's5_features']
	validation set:  s2_features

	Confusion matrix
	[[ 194.    8.]
	 [   5.  191.]]
	Recall:    0.974874371859
	Precision: 0.960396039604
	F1 Score:  0.967581047382
	----------------------------------------

	loading feature file ./data/s1_features
	loading feature file ./data/s2_features
	loading feature file ./data/s5_features
	loading feature file ./data/s3_features
	Calculating the loss function value on the validation set ...
	Epoch 0: average loss function value on validation set: 0.362763
	Convergence critera: when difference > -0.005000
	Epoch 10: average loss function value: 0.276761  |  Difference: -0.086001
	Epoch 20: average loss function value: 0.271401  |  Difference: -0.005360
	Epoch 30: average loss function value: 0.269151  |  Difference: -0.002249
	loading feature file ./data/s4_features

	----------------------------------------
	test set         s4_features
	training sets: ['s1_features', 's2_features', 's5_features']
	validation set:  s3_features

	Confusion matrix
	[[ 195.   13.]
	 [   4.  186.]]
	Recall:    0.979899497487
	Precision: 0.9375
	F1 Score:  0.958230958231
	----------------------------------------

	loading feature file ./data/s1_features
	loading feature file ./data/s2_features
	loading feature file ./data/s3_features
	loading feature file ./data/s4_features
	Calculating the loss function value on the validation set ...
	Epoch 0: average loss function value on validation set: 0.361495
	Convergence critera: when difference > -0.005000
	Epoch 10: average loss function value: 0.277648  |  Difference: -0.083848
	Epoch 20: average loss function value: 0.272506  |  Difference: -0.005142
	Epoch 30: average loss function value: 0.270428  |  Difference: -0.002078
	loading feature file ./data/s5_features

	----------------------------------------
	test set         s5_features
	training sets: ['s1_features', 's2_features', 's3_features']
	validation set:  s4_features

	Confusion matrix
	[[ 192.    2.]
	 [   7.  196.]]
	Recall:    0.964824120603
	Precision: 0.989690721649
	F1 Score:  0.977099236641
	----------------------------------------

	========================================
	Averages of 5-fold cross validation
	Recall:    0.968844221106
	Precision: 0.96434999261
	F1 Score:  0.966456061963
	========================================

	real	0m38.678s
	user	0m38.432s
	sys	0m0.096s


# License

MIT License

Copyright (c) Kerry Zhang

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

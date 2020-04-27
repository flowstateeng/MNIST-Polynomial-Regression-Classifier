<p align="center">
  <img width='650' src='https://github.com/chivington/MNIST-Polynomial-Regression-Classifier/blob/master/imgs/mnist.png' alt='MNIST Digits'/>
</p>



# MNIST-Polynomial-Regression-Classifier
This is a Polynomial Regression model that learns to classify hand-written digits from the MNIST dataset. Try different values for 'lambda' and 'p' to experiment with output.

![Build Status](https://img.shields.io/badge/build-Stable-green.svg)
![License](https://img.shields.io/badge/license-NONE-green.svg)
<br/><br/><br/>

## Contents
* [Prerequisites](https://github.com/chivington/MNIST-Polynomial-Regression-Classifier/tree/master#prerequisites)
* [Installation](https://github.com/chivington/MNIST-Polynomial-Regression-Classifier/tree/master#installation)
* [Testing](https://github.com/chivington/MNIST-Polynomial-Regression-Classifier/tree/master#testing)
* [Usage](https://github.com/chivington/MNIST-Polynomial-Regression-Classifier/tree/master#usage)
* [Authors](https://github.com/chivington/MNIST-Polynomial-Regression-Classifier/tree/master#authors)
* [Contributing](https://github.com/chivington/MNIST-Polynomial-Regression-Classifier/tree/master#contributing)
* [Acknowledgments](https://github.com/chivington/MNIST-Polynomial-Regression-Classifier/tree/master#acknowledgments)
* [License](https://github.com/chivington/MNIST-Polynomial-Regression-Classifier/tree/master#license)
<br/>

## Prerequisites
  * Python
  * Numpy
  * Matplotlib
<br/><br/>


## Installation
```bash
  git clone https://github.com/chivington/MNIST-Polynomial-Regression-Classifier.git
```

## Testing
Change into the directory and run:
```bash
  python mnist_polynomial_regression_classifier.py
```

<br/>

## Usage
Open the file 'mnist_polynomial_regression_classifier.py' and edit the hyperparameters 'lambda' (named "lambd" in the code since "lambda" is a reserved keyword in the Python programming language) and the vector 'p' to experiment with finding optimal values for accuracy and quick convergence. The program will:

1. Load the MNIST dataset.
2. Split it into "training," "validation," and "testing" sets.
3. Display a random digit from the training set.
4. Train the model on the various p-values, displaying the training error, test error and training time.
5. Display a plot of the training errors, validation errors, training times, with respect to the various p-values.
6. Display a digit from the test set, along with it's classification and label.
7. Print out the final training and test set errors to the terminal.
8. End.

Large values of 'p' will result in increased training times. For my Surface Book (2.81ghz i7, 8gb ram, GeForce 940m), a p-value of 7500 takes ~374 seconds to train and achieves ~4.48% testing error rate, as shown in the images below. The datasets needed to train and make predictions are included and you should not have to move or rename any files or filenames in the code.

<p align="center">
  <img width='650' src='https://github.com/chivington/MNIST-Polynomial-Regression-Classifier/blob/master/imgs/random-img.jpg' alt='Random Digit'/>
</p>

<p align="center">
  <img width='650' src='https://github.com/chivington/MNIST-Polynomial-Regression-Classifier/blob/master/imgs/errors-and-times.jpg' alt='Training & Validation Errors'/>
</p>

<p align="center">
  <img width='650' src='https://github.com/chivington/MNIST-Polynomial-Regression-Classifier/blob/master/imgs/classification.jpg' alt='Classification Test'/>
</p>

Feel free to ask me questions on [GitHub](https://github.com/chivington) or [LinkedIn](https://www.linkedin.com/in/johnathan-chivington/)
<br/><br/>


## Authors
* **Johnathan Chivington:** [GitHub](https://github.com/chivington) or [LinkedIn](https://www.linkedin.com/in/johnathan-chivington/)

## Contributing
Not currently accepting outside contributors, but feel free to use as you wish.

## License
There is no license associated with this content.
<br/><br/>

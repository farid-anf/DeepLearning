# DeepLearning

## Gradient Descent 


Gradient descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of its steepest descent, which is determined by the negative of the gradient.

### Mathematical Explanation

#### Objective Function
Suppose we want to minimize a function  $f(\boldsymbol{\theta})$, where  $\boldsymbol{\theta}$ is a vector of parameters. The goal is to find the value of $\boldsymbol{\theta}$ that minimizes $f(\boldsymbol{\theta})$.

#### Gradient
The gradient of $f$ at $\boldsymbol{\theta}$, denoted by $\nabla f(\boldsymbol{\theta})$, is a vector of partial derivatives:


$$\nabla f(\boldsymbol{\theta}) = \left[ \frac{\partial f}{\partial \theta_1}, \frac{\partial f}{\partial \theta_2}, \ldots, \frac{\partial f}{\partial \theta_n} \right]^T.$$

The gradient points in the direction of the steepest increase of $f$.

#### Update Rule
Gradient descent updates $\boldsymbol{\theta}$ iteratively:

$$
\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \eta \nabla f(\boldsymbol{\theta}^{(t)}),
$$

where:
- $t$ is the iteration index,
- $\eta$ is the learning rate, a small positive scalar that determines the step size.

#### Intuition
- The term $\nabla f(\boldsymbol{\theta}^{(t)})$ gives the direction of steepest ascent. To minimize $f$, we move in the opposite direction, $-\nabla f(\boldsymbol{\theta}^{(t)})$.
- The learning rate $\eta$ controls how far we move along this direction in each iteration. A small $\eta$ ensures stability but may converge slowly, while a large $\eta$ may overshoot the minimum or diverge.

#### Convergence
Gradient descent converges to a local minimum if:
1. $f(\boldsymbol{\theta})$ is differentiable,
2. $\eta$ is chosen appropriately,
3. The gradient $\nabla f(\boldsymbol{\theta})$ does not vanish.

#### Example
Consider a simple quadratic function:

$$
f(\theta) = \frac{1}{2} \theta^2.
$$

The gradient is:

$$
\nabla f(\theta) = \frac{\partial f}{\partial \theta} = \theta.
$$

The update rule becomes:

$$
\theta^{(t+1)} = \theta^{(t)} - \eta \theta^{(t)}.
$$

After $t$ iterations:

$$
\theta^{(t)} = (1 - \eta)^t \theta^{(0)}.
$$

As $t \to \infty$, $\theta^{(t)} \to 0$, provided $0 < \eta < 2$).

This illustrates how gradient descent converges to the minimum $\theta = 0$.

## Course 1. Neural Networks and Deep Learning 



### Logistic Regression Overview

Logistic regression is a supervised learning algorithm used for binary classification problems. It predicts the probability that a given input belongs to a particular class. The output is a value between 0 and 1, which is interpreted as the probability of the positive class.

---

### Mathematical Foundation

#### 1. **Logistic Function (Sigmoid Function)**
The logistic regression model uses the **sigmoid function** to map the linear regression output to probabilities. The sigmoid function is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Where $z = \mathbf{w}^\top \mathbf{x} + b$, which is the linear combination of input features ($\mathbf{x}$), weights ($\mathbf{w}$), and bias ($b$).

---

#### 2. **Prediction**
The model predicts probabilities:

$$P(y=1 | \mathbf{x}) = \sigma(z)$$

The decision boundary is typically set at 0.5. If $P(y=1 | \mathbf{x}) \geq 0.5$, the model predicts $y=1$; otherwise, it predicts $y=0$.

---

#### 3. **Cost Function**
Logistic regression uses the **log-loss** (cross-entropy loss) as its cost function:

$$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

Where:
- $m$: number of training examples
- $y^{(i)}$: actual label (0 or 1) for the \(i\)-th example
- $\hat{y}^{(i)} = \sigma(\mathbf{w}^\top \mathbf{x}^{(i)} + b)$: predicted probability

---

#### 4. **Gradient Descent**
To minimize the cost function, gradients with respect to $\mathbf{w}$ and $b$ are computed:

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right) x_j^{(i)}
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)
$$

---

### Implementation in Python

Here’s how to implement logistic regression using Python from scratch and with libraries.

---

#### 1. **From Scratch**

```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.iterations):
            model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(model)

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(model)
        return [1 if i > 0.5 else 0 for i in y_predicted]
```

---

#### 2. **With scikit-learn**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```

---

### Key Points to Remember
1. **Advantages**: 
   - Simple and interpretable.
   - Works well for linearly separable data.
   - Outputs probabilities.

2. **Limitations**:
   - Assumes linear relationship between input features and log-odds.
   - Struggles with non-linear patterns (can be mitigated with feature engineering or kernel methods).

3. **Best Practices**:
   - Feature scaling improves convergence.
   - Regularization (L1/L2) helps with overfitting.

Let me know if you’d like to dive deeper into specific concepts!









## Course 2. Improving Deep Neural Networks: Hyperparameter tuning, Regularization, and Optimization 
## Course 3. Structuring your Machine Learning project 
## Course 4. Convolutional Neural Networks 
## Course 5. Natural Language Processing: Building sequence models

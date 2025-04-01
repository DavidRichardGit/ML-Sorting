# ML-Sorting :1234:
Predicting the optimal sorting algorithm for specific arrays with given presortedness metrics :chart_with_upwards_trend:

Results:
- [Predicting Algorithms given all presortedness metrics of full array](https://github.com/DavidRichardGit/Presortedness-ML/blob/main/MLModels/NNModel_fullArray.ipynb)
- [Predicting Algorithms minimizing amount of comparisons needed for sorting + presortedness calculation](https://github.com/DavidRichardGit/Presortedness-ML/blob/main/MLModels/NNModel_dynamic%20(1).ipynb)

## Generic Workflow :clipboard:

In order to have a model predict a comparison-optimal sorting algorithm for not fully random data we have to:

### 1. Sorting algorithm selection
We have to chose a number of decent sorting algorithms that can all be good in different situations. The more clear cut their strengths are, the easier it is for the model to select the right algorithm resulting in a higher prediction accuracy.

### 2. Data
Next we have to acquire large quantities of data. It is beneficial for us if the data has a lot of characteristics to it that can be predicted by only looking at a few samples in the data partition. Therefore the data cannot be fully random. A datascience data collection is ideal for this task.

### 3. Feature engineering
For the model to efficiently classify the data we need a way to summarize the data into information-rich metrics. Presortedness metrics are very suitable for the characterization of the data sortedness.

### 4. Model
After the creation of a training set, we have to chose a machine learning model. Neural networks fit the requirements, because they are good at recognizing complex relationships and are flexible enough.

### 5. Optimization
Finally we can get to optimizing all the variables influencing the final model. The most important ones are:

	- sampling strategy
		how the numbers are chosen that we analyse
	- presortedness combinations
		which presortedness metrics do we analyse
	- sampling size
		how many numbers do we analyse
	- hyperparameters
		how many layers, with what layersize do we chose

Optimizing everything at once would be too computationally expensive. That is why we had to optimize them one by one with a strategic approach.

## Optimality of a sorting algorithm :100:

The efficiency of a sorting algorithm can be captured from multiple different angles. The fastest might not always be the first choice, since there are other important measures of an algorithm, such as memory complexity, stability or wheter an algorithm can handle a constant stream of input or not. The time complexity is hard to inspect, because there are many factors playing into it: the type of data, the size of the data, the cost of comparing and swapping values, the hardware etc. Another aspect which makes it hard to investigate the time complexity is the implementation of the algorithm. Comparing any self-implemented sorting algorithm (can be the most efficient one) in python to the .sort() function that uses timsort does not make sense, since timsort will outperform by a large margin. This is due to the fact that the .sort() function is written in C, highly optimized and doesn't have to be interpreted line-by-line (see [here](https://github.com/DavidRichardGit/Presortedness-ML/blob/main/pythonTimsort%20(2).ipynb)).

This is why i will be focusing on the comparison complexity, it is not dependent on the hardware and the implementation. Additionally it is a good predictor of the computational complexity of an algorithm.

## Dataset properties :bar_chart:

In order to explore the relationship between array formation and algorithm efficency, a wide variety of data is needed. Working with the [Kaggle Api](https://www.kaggle.com/docs/api) gives us a way to fetch realistic data in a fast and automated way.

- [Data Downloading, Importing and Engineering Insights](https://github.com/DavidRichardGit/Presortedness-ML/tree/main/Data)

Taking Data from a Datascience platform has the advantage that the columns are not arranged in a random way. For instance: Numbers can correlate with other sorted columns, which can make them "almost" sorted. This results in diverse ranges and combinations of presortedness kinds. To be able to characterize different kinds of presortedness we have to define it formally:

## Presortedness :1234:

For a sorting algorithm to be considered optimal for a presorted array the algorithm has to be adaptive. Adaptive means that the algorithm performs asymptotically optimal on an already sorted array. 
Arrays of numbers can be sorted or almost sorted in different ways. Measures of presortedness helps us describe how sequences are arranged to get insight about the data.
Formally the following conditions formulated by Marcello La Rocca and Domenico Cantone in _NeatSort, a practical adaptive algorithm_ have to be satisfied: 

Given two sequences X and Y of distinct elements, a metric of presortedness M is a function that satisfies the following axioms:

* If X is sorted, then M(X) = 0
* If X and Y are order isomorphic, then M(X) = M(Y)
* If X is a subset of Y, then M(X) ≤ M(Y)
* If every element of X is smaller than every element of Y, then M(X.Y) ≤ M(X) + M(Y)
* M({x}.X) ≤ |X| + M(X) for every natural integer X

Similarly to the general adaptivity of an algorithm an adaptive algorithm with respect to a certain measure of presortedness is defined as follows by Svante Carlsson, Christos Levcopoulos and Ola Petersson in _Sublinear Merging and Natural Mergesort_:

*A sorting algorithm is said to be adaptive with respect to a measure of presortedness if it sorts all sequences, but performs particularly well on those having a high degree of presortedness according to the measure.*

Some relation of presortedness metrics can be visualized by this graph from O. Petersson and A. Moffat in _A framework for adapative sorting_. The metric on the upper node is algorithmically finer than the one connected lower node. If a metric f<sub>1</sub> is algorithmically finer than another metric f<sub>2</sub> it means, that an f<sub>1</sub> optimal sorting algorithm is automatically f<sub>2</sub> optimal.

![Presortedness graphic](https://i.imgur.com/SnTLZeh.png)

To get the maximum amount of information about the way an array is arranged it makes sense to have a well distributed set of metrics.

### REM(X) / Number of Deletions :x:
Minimum number of elements that need to be removed from an array to obtain a sorted sequence. It can be: |X| - 1 if the sequence is arranged in a reversed order.

```python
def deletions(arr):
    def ceil_index(sub, val):
        l, r = 0, len(sub)-1
        while l <= r:
            mid = (l + r) // 2
            if sub[mid] >= val:
                r = mid - 1
            else:
                l = mid + 1
        return l
 
    sub = [arr[0]]
    for i in range(1, len(arr)):
        if arr[i] >= sub[-1]:
            sub.append(arr[i])
        else:
            sub[ceil_index(sub, arr[i])] = arr[i]
 
    return len(arr) - len(sub)
```

### RUNS(X) / Number of Runs :runner:
Minimum number of increasing sequences in an array minus one. |X| - 1 when X is arranged in reverse order.

```python
def runs(arr):
    count = 0

    for key in range(1,len(arr)):
        if arr[key] < arr[key-1]:
            count += 1

    return count
```

### INV(X) / Number of Inversions :left_right_arrow:
Minimum number of inversions in X, where one inversion is one pair of elements that are not in order. It can be: |X| * (|X| - 1) / 2 when X is arranged in reverse order.

```python
def inversions(arr):
    count = 0

    for key in range(len(arr)):
        for j in range(key):
            if arr[key] < arr[j]:
                count += 1

    return count
```

### DIS(X) / Maximum Distance of inversion :repeat:
Maximum distance of an inversion, where one inversion is one pair of elements that are not in order. It can be: |X| - 1 when the last element of X is smaller than the first element of X.

```python
def max_dist_inversion(arr):
    c_max_dist = 0

    for key in range(len(arr)):
        for j in range(key):
            if arr[key] < arr[j]:
                c_max_dist = max(key-j,c_max_dist)

    return c_max_dist
```

### INV_DIS(X) / Number of Inversions combined with Maximum Distance of inversion :left_right_arrow: :repeat:
Minimum number of inversions in X and maximum distance of an inversion can easily be combined:

```python
def inv_dis(arr):
    c_max_dist = 0
    inv = 0
    
    for key in range(len(arr)):
        for j in range(key):
            if arr[key] < arr[j]:
                c_max_dist = max(key-j,c_max_dist)
                inv += 1

    return inv, c_max_dist
```

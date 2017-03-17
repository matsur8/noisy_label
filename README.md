# noisy_label

code to learn from noisy labels   
An EM algorithm is used to deal with unobserved true labels [1].  
(There are some differences from [1]. See source codes.)

## demo
test data (noise-free)  
<img src="https://github.com/matsur8/noisy_label/blob/master/result/symmetric/test_data.png" width="320" height="200" />

### symmetric noise
p(y_noisy=1| y_clean=0) = p(y_noisy=0| y_clean=1) = 0.3


<img src="https://github.com/matsur8/noisy_label/blob/master/result/symmetric/training_data.png" width="320" height="200" /><img src="https://github.com/matsur8/noisy_label/blob/master/result/symmetric/sigmoid.png" width="320" height="200" />

### asymmetric noise
p(y_noisy=1| y_clean=0) = 0.05, p(y_noisy=0| y_clean=1) = 0.4

<img src="https://github.com/matsur8/noisy_label/blob/master/result/asymmetric/training_data.png" width="320" height="200" /><img src="https://github.com/matsur8/noisy_label/blob/master/result/asymmetric/sigmoid.png" width="320" height="200" />


## references
[1] Rayker et al., "Learning From Crowds", 2010, http://www.jmlr.org/papers/volume11/raykar10a/raykar10a.pdf

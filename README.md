
# SPAM CLASSIFIER USING LOGISTIC REGRESSION 
The project focusses on building a spam classifier using the popular sms data set from the uci machine learning repository.
concepts of NLP,classfication using logistic regressing are implemented to build this project. 
# contents
1. [concept of the project](##CONCEPT OF THE PROJECT-2)
2. [THE DATASET](##THE DATASET-2)
3. [implementing natural language processing](##heading-3)
4. [
## CONCEPT OF THE PROJECT
spams are any unwanted or unnecessary messages we get. They include scams, virus links and advertisements and these are different from the messages we get from friends, family, college , workplace etc. 
- #### why do we need to classify and delete spams?
we need to classify and delete spams because they may sometimes contain  viruses or trojans which may get installed to the system once the mail is opened.
we can implement **classification techniques in machine learning** to classify messages which are spam from those which are not. 
- #### algorithms and techniques implemented in this project
  - **LOGISTIC REGRESSION**
  - **NATURAL LANGUAGE PROCESSING**
 - #### python libraries used in this project
   - **numpy**
   - **pandas**
   - **scikit learn**
   - **NLTK**
   - **regular expression**
   - **matplotlib**
## THE DATASET
the dataset used in this project is [**SMS SPAM COLLECTION DATA SET**](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection). it is quiet a popular dataset in the world of machine learning. it contains totally 5574 instances where the first word of every instance tells if the message is spam or ham. The message and the label are seperated by a tab space.

![](https://github.com/uday1and100/machine-learning-and-deep-learning-projects/blob/master/Untitled.jpg)

we can say that we have two features namely the labels which tells if the message is spam or ham and the messages itself. this data is used to train the logistic regression model

## implementing NATURAL LANGUAGE PROCESSING
the obtained data is first seperated into two columns, labels and messages. 
now the data set has two features labels and messages where labels tell if the corresponding image is spam or not
 - #### text mining
   the following techniques are implemented on the data
     - **tokenization**
     - **lemmitization** 
     - **removing stopwords**
     
  tokenization is done on the sentence which yields in the sentence being split into individual words which if then followed by lemmitization.
  regular expression techniques are then implemented to remove the punctuaction marks. stop words(for ,is, the etc) are then removed from the data.
  this results in a list which contains sentences without any stopwords and punctuation marks.

 - #### bag of words using count vectorizer
the mined data is then fed to the count vectorizer which creates the bag of words. the bag of words is nothing but an array with numbers which tells how many times a particular word has appeared in the sentence. the words are mapped to columns and the sentences are mapped to columns.this creates an array wtih numbers . 

![](https://image.slidesharecdn.com/wordembedings-whythehype-151203065649-lva1-app6891/95/word-embeddings-why-the-hype-4-638.jpg?cb=1449126428)

## FEATURE ENGINEERING FOR THE IMBALANCED DATASET
the dataset is imbalanced. there are more number of instances for hams when compared to spams. this may to lead a bias in the model. so we create more instances for the "spam" category using oversampling. this can be done using imblearn library and we make equal number for instances for both.
we then map spam=1, and ham = 0 in the dataset. thus the dataset is ready to be fed to the algorithm.

## CREATING THE MODEL
  the algorithm used here is LOGISTIC REGRESSION which is a pretty good algorithm for binary classification. I chose this algorithm as it is simple and i can obtain the    probabilites of predictions easily which makes it easier for me to vary the thresholds and check for precision and recall. 
 the following steps are done
   - the model is imported
   - the data is split in the ratio of 80% training data and 20% test
   - the training data is fit to the model
## PREDICTION AND METRICS 
 we use the test data to predict the probability of each instance. we do this so that we can try for different thresholds for classification. the default threshold for logistic regression is 0.5 i.e probability above 0.5 means it is a spam and below it means it is not a spam
 we try for [0.5,0.6,0.65,0.7]. 
 
 - #### why higher thresholds?
   we test for higher thresholds because we focus on detecting the spam but at the same time we dont want normal messages to get classfied as spams. this may lead to important messages getting thrown to the spam folder. we focus on
   - **REDUCING THE SENSITIVITY**
   - **IMPROVING THE SPECIFICITY**
   this means that 
   - **there will be very less FALSE POSITIVE CASES**(normal mails getting classified as spams)
 we cannot improve both sensititivty and specificity together. there is a trade off. 
 
 - #### measuring sensitivity and specificity
   **RECALL** is used to measure **SENSITIVITY**
   **PRECISION** is used to measure **SPECIFICITY**
   for various thresholds we plot the PRECISION VS RECALL CURVE. 
   
 - #### why precision vs recall curve and not the ROC CURVE?
   the general rule of thumb is that when we are concerned about the sensitivity we plot the ROC CURVE. when we are concerned about specificity we plot the precision vs recall curve. roc curve helps us understand senstivity since it focusses on true positive rate vs false positive rate. we are concerned about specificity in this case which is why we use the precision vs recall curve
    
   ![](https://github.com/uday1and100/Spam-classifier-using-logistic-regression/blob/master/acc.jpg)
   
the graph is plotted for the thresholds [0.5,0.6,0.65,0.7] 
for the thresholds [0.5,0.6] the sensitivity is very high(99%) which may lead to false positives. for the threshold 0.7, the sensiitivity drops to 87%. the ideal threshold would be **0.65** where the precision is 96% and the sensitivity is also 98%.

## INFERENCE
the project was successfully built using the required libraries. by varying the thresholds , the best values of accuracy, precision and recall was obtained.
**accuracy** = 97%
**precision** = 96%
**recall** = 98%
   
   
   

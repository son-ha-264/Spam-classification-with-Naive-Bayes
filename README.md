# Spam Classification with Naive Bayes model

The [UCI Spam Colletion Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) is a dataset of labelled SMS colleted for mobile phone spam research. The SMS texts mostly come from UK and Singapore, and are collected from various free sources on the Internet. In this notebook I will build a simple Naive Bayes model to classify which SMS is spam and which is not. This is the classic 'Ham (not spam) or Spam' task in Natural Language Processing (NLP).

## 1. Load the data
![](figures/table_1.png)
<p align="center"> 
  Table 1: First 10 rows of the table 
</p>

![](figures/table_2.png)
<p align="center"> 
  Table 2: Summary table
</p>

According to the summary table, only around 13.4% of the observations are spams. In addition, there are some duplicate rows. This might be because of the SMS texts were collected not from one source and there were overlaps between the sources. We will remove the duplicates. This still leaves us with a lot of data to work with.

![](figures/hist_1.png)

The charts above are the histograms of spam(right) and ham(left). As we can see, spam texts, which averages around 150 characters, are generally longer than ham, whose average is much lower than that. This discovery will play a part later in our analysis.

## 2. Text Preprocessing

For our Naive Bayes algorithm to understand the data we have to find a way to convert these texts into vectors. The most popular way is the **Bag-of-Words** (BoW) model. How it works is that we create a 'vocabulary' which contains all the words in our data, and assign each word in that vocabulary an index. Then each SMS text can be represented by a vector whose length is the size of that vocabulary, and each of its entries is the count/frequency of the word at that index. [Wikipedia] (https://en.wikipedia.org/wiki/Bag-of-words_model) has some nice examples of BoW.

Before turning our texts into a vocabulary, let's clean and tokenise the text. For each text, I perform the following steps:
- (1) Remove all punctuations, standalone (e.g. full stops) or as a part of a word (e.g. What's)
- (2) Remove stopwords. They are words that does not contribute much meaning to the sentences, thus is safe to remove (e.g. it's, hers,him,...).
- (3) Change all words to lowercase for consistency. This is important since the capitalisation of words are all over the place in SMS texts.
- (4) Remove all non-alphabetic entries 

I have also tried:
- (5) Remove 1-letter words, again since they don't convey much meaning
- (6) Apply [Porter Stemmer](https://tartarus.org/martin/PorterStemmer/).

But they do not result in any improvement in performance so I drop them. However it is always good practice to try everything and then decide which course of action is the best for model performance.

A sample text ''

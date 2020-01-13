# Multi-Label Text Classification using Long Short Term Memory (LSTM) neural network architecture

In this project, I have implemented LSTM neural network architecture to classify movies into 12 different genres based on their plot summaries. 
Each movie can be tagged with either one or more genres (for eg. a movie can be tagged as only "Romantic" or both "Romantic" and "Comedy").


Files:
	
	- 1_DataCleaning.ipynb : Code to clean up messy genres, keep only top 12 most occuring genres and merge the plots dataset with the movie metadata dataset

	- 2_Mainfile.ipynb : Code for creating word embeddings, implementation of LSTM and model evaluation
	
	- GenreMapping.csv : Manually mapped genre values

	- movie_metadata.tsv : Contains metadata for different movies

	- plot_summaries.txt : Plot Summaries for different movies

	- glove.6B.100d.txt : 100-dimensional GloVe Embedding values for 6 billion words


The datasets were obtained from the UCI Machine Learning Repository. 


Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. 
Unlike standard feedforward neural networks, LSTM has feedback connections. 
It can not only process single data points (such as images), but also entire sequences of data (such as speech or video). 
For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition, speech recognition and anomaly detection in network traffic or IDS's (intrusion detection systems).


Part 1	- Data Cleaning

		- Pre-process messy unstructured text by removing accents, punctuations, converting tokens to lower case, removing a customized set of stop words, etc. to get better prediction results

		- Clean up genres: For e.g. Separate “Romcom” into “Romantic” and “Comedy”, combine similar genres into one common genre

		- Filter out the top 12 most occurring genres to get the best possible prediction accuracy

		- One hot-encode genres for each movie so that it can be fed into a neural network


Part 2	- Word Embeddings using GloVe

		Word Embedding is method used to create dense vector representation for each word in the corpus. GloVe is an unsupervised learning algorithm for obtaining vector representations for words. 
		Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. 


Part 3	- Using LSTM for multi-label genre classification for movies. Implemented shallow neural network with 4 layers:

		- Input Layer

		- Embedding layer

		- LSTM Layer

		- Output Layer


Part 4	- Train and evaluate model

		- The model achieved an accuracy of 88.1% on the test data set


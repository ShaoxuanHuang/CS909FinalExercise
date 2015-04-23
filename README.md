# CS909FinalExercise
This is the repository of CS909 final coursework.
The codes of coursework is written by R

1. The FinalExerciseCode.R file includes all codes and all of these have been tested in following environment:
R version 3.1.2
-RStudio Version 0.98.1062

2. The R packages we need:
-package ‘tm’ version 0.6
-package ‘topicmodels’ version 0.2-1
-package ‘e1071’ version 1.6-4
-package ‘randomForest’ version 4.6-10
-package ‘fpc’ version 2.1-9
-package ‘cluster’ version 1.15.3

3. It is easy to run these codes, we just need to follow the annotation in FinalExerciseCode.R file.

4. The codes are structured as following:
	\-Task 1
	    \-1.1 Cleaning part
            \-1.2 Pre-processing part
	\-Task 2
	    \-2.1 to get LDA features
	    \-2.2 to get TF*IDF features
	\-Task 3
	    \-3.1 use LDA as features
	        \-classification
		    \-SVM
		    \-naiveBayes
		    \-randomForest
		\-evaluation
	   	    \-SVM
		    \-naiveBayes
		    \-randomForest
	    \-3.2 use TF*IDF as features
			…
			…
			…
	    \-3.3 use LDA+TF*IDF as features
			…
			…
			…
	\-Task 4
	    \-4.1 Clustering
		\-K-means
		\-Hierarchical Agglomerative
		\-DBSCAN
	    \-4.2 evaluation part
		\-K-means
		\-Hierarchical Agglomerative
		\-DBSCAN
 
5. The dataset folder contains the data saved during coding.

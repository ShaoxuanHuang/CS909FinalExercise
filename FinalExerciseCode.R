#==========#
#  Task 1  #
#==========#

setwd("~/Desktop/Term2/CS909DM/exercise/week10/reuters")
reutersCSV<-read.csv(file="reutersCSV.csv",header=T,sep=",")

#remove the rows without any text
tempRowT<-c()
for(i in 1:nrow(reutersCSV)){
    if(reutersCSV[i,139]=='' || reutersCSV[i,140]=='')
        tempRowT<-c(tempRowT,i)
}
reutersRemove<-reutersCSV[-tempRowT,]
#tempRowT=2535
#so we removed 2535 rows

#remove the rows that all topics are 0
tempRowZ<-c()
for(i in 1:nrow(reutersRemove)){
    if(sum(reutersRemove[i,4:138])==0)
        tempRowZ<-c(tempRowZ,i)
}
reutersRemove<-reutersRemove[-tempRowZ,]
#tempRowZ=8693
#so we removed 8693 rows

#remove the columns that all topics are 0
tempColZ<-c()
for(i in 4:138){
    if(sum(reutersRemove[1:ncol(reutersRemove),i])==0)
        tempColZ<-c(tempColZ,i)
}
reutersRemove<-reutersRemove[,-tempColZ]
#tempColZ=90
#now we have reutersRemove, which has 10350 obs and 50 variables

write.csv(reutersRemove, "reutersRemove.csv", row.names = F)


#Pre-process part#
#================#

library(tm)

#convert to Corpus
reutersCorpus<-Corpus(VectorSource(reutersRemove[,50]))

reutersCorpus<-tm_map(reutersCorpus,removeWords, stopwords("english"))  #remove stopwords
reutersCorpus<-tm_map(reutersCorpus,removePunctuation)                  #remove punctuation
reutersCorpus<-tm_map(reutersCorpus,removeNumbers)                      #remove numbers
reutersCorpus<-tm_map(reutersCorpus,tolower)                            #convert all text to lowercase
reutersCorpus<-tm_map(reutersCorpus,stripWhitespace)                    #remove extra whitespace
reutersCorpus<-tm_map(reutersCorpus,PlainTextDocument)                  #convert corpus to plain text





#==========#
#  Task 2  #
#==========#

library(topicmodels)

#to get LDA features
dtm <- DocumentTermMatrix(reutersCorpus, control = list(wordLengths = c(3,Inf)))
dtmS<-removeSparseTerms(dtm,sparse=0.95)
reutersLDA<-LDA(dtmS,10, method = "Gibbs")
LDAterms<-as.vector(terms(reutersLDA, 50))
LDATDM<- DocumentTermMatrix(reutersCorpus,control = list(wordLengths = c(3,Inf),dictionary= LDAterms))
LDATDMf<-as.data.frame(inspect(LDATDM))
#10350 obs and 168 var

write.csv(LDATDMf, "LDATDMf.csv", row.names = F)


#to get TF*IDF features
reutersTFIDF<-weightTfIdf(dtm)
reutersTFIDFS<-removeSparseTerms(reutersTFIDF,sparse=0.95)
TFIDFf<-as.data.frame(inspect(reutersTFIDFS))
#10350 obs and 168 var

write.csv(TFIDFf, "TFIDFf.csv", row.names = F)





#==========#
#  Task 3  #
#==========#

reutersRemove<-read.csv(file="reutersRemove.csv",header=T,sep=",")
LDATDMf<-read.csv(file="LDATDMf.csv",header=T,sep=",")
TFIDFf<-read.csv(file="TFIDFf.csv",header=T,sep=",")



#extract top 10 topics index
topicNames<-c("topic.earn", "topic.acq", "topic.money.fx", "topic.grain", "topic.crude", "topic.trade", "topic.interest", "topic.ship", "topic.wheat", "topic.corn")

TempT<-c()
for(i in 1:10){
    for(j in 4:48){ #topic areas
        if(names(reutersRemove[j])==topicNames[i])
            TempT<-c(TempT,j)
    }
}

reutersTop<-cbind(reutersRemove[,c(3,TempT)])
#select top10 and purpose column and save as a new dataset

write.csv(reutersTop, "reutersTop.csv", row.names = F)



#3.1 use LDA as features#
#=======================#

reutersLDAf<-cbind(reutersTop,LDATDMf)
#add a column to represent topics

reutersLDAf[,"topic"]<-"empty"

# 1-purpose
# 2:11-Top10 topics
# 12:179-LDA features
# 180-topic

#strategy for the documents assigned multiple topics: 
#assign only the minority class
for(i in 1:nrow(reutersLDAf)){
    for(j in 2:11){
        if(reutersLDAf[i,j]==1)
            reutersLDAf[i,180]<-colnames(reutersLDAf)[j]
    }
}

#save the topic as backup
topicBackup<-reutersLDAf[,180]

##remove the rows that all topics are 0
tempNA<-c()
for(i in 1:nrow(reutersLDAf)){
    if(reutersLDAf[i,180]=="empty")
        tempNA<-c(tempNA,i)
}
reutersLDAf<-reutersLDAf[-tempNA,]
#now has 8599 obs
#tempNA could use later

#split Train and Test dataset
LDATrain<-reutersLDAf[which(reutersLDAf$purpose=="train"),]
LDATest<-reutersLDAf[which(reutersLDAf$purpose=="test"),]

#remove useless column
LDATrain<-LDATrain[,-c(1:11)]
LDATest<-LDATest[,-c(1:11)]

write.csv(LDATrain, "LDATrain.csv", row.names = F)
write.csv(LDATest, "LDATest.csv", row.names = F)



library(e1071)
library(randomForest)

#begin to model and predict#
#++++++++++++++++++++++++++#

#SVM
SVMLDATrain<-LDATrain
SVMLDATest<-LDATest

SVMLDAModel <- svm(topic ~ .,SVMLDATrain)
SVMLDAPre <- predict(SVMLDAModel,SVMLDATest)
SVMLDATable <- table(SVMLDAPre, SVMLDATest$topic)

#SVMLDAPre        topic.acq topic.corn topic.crude topic.earn topic.grain topic.interest topic.money.fx topic.ship topic.trade topic.wheat
#  topic.acq            601          5          19         21           4              3             10         13           3           1
#  topic.corn             0         19           1          0           4              0              0          2           3           7
#  topic.crude            1          3          82          0           4              2              0          9           0           2
#  topic.earn            27          9           9       1018           4             18             13          6           9           7
#  topic.grain            0          0           0          0           1              0              0          0           0           0
#  topic.interest         0          1           1          1           0             63             14          0           0           0
#  topic.money.fx         2          0           1          1           1              8             50          0           4           0
#  topic.ship             2          2           8          0           2              3              7         46           2           4
#  topic.trade            1          1           4          0           4              5              5          6          83           3
#  topic.wheat            1          8           0          0           6              0              0          1           3          24



#naiveBayes
NBLDATrain<-LDATrain
NBLDATest<-LDATest

NBLDAModel <- naiveBayes(topic ~ .,NBLDATrain)
NBLDAPre <- predict(NBLDAModel,NBLDATest)
NBLDATable <- table(NBLDAPre, NBLDATest$topic)

#NBLDAPre         topic.acq topic.corn topic.crude topic.earn topic.grain topic.interest topic.money.fx topic.ship topic.trade topic.wheat
#  topic.acq            436          0           7         23           0              1              0          0           0           0
#  topic.corn             1         11           3          1           2              0              0          2           6           1
#  topic.crude           27          2          33         31           0              0              0          2           1           0
#  topic.earn             9          0           0        925           1              0              0          0           0           1
#  topic.grain           31          6          30         41           4              2             10         10          12           3
#  topic.interest         3          2           1          2           0             57             34          0          11           1
#  topic.money.fx         2          1           1          0           1             23             29          0           8           0
#  topic.ship            17          0          13          1           0              4              6         16           3           2
#  topic.trade            2          2           3          2           0              4              4          4          34           0
#  topic.wheat          107         24          34         15          22             11             16         49          32          40



#randomForest
RFLDATrain<-LDATrain
RFLDATest<-LDATest

RFLDAModel <- randomForest(topic ~ .,RFLDATrain)
RFLDAPre <- predict(RFLDAModel, RFLDATest)
RFLDATable <- table(RFLDAPre, RFLDATest$topic)

#RFLDAPre         topic.acq topic.corn topic.crude topic.earn topic.grain topic.interest topic.money.fx topic.ship topic.trade topic.wheat
#  topic.acq            602          5          12         19           1              3              6          6           3           2
#  topic.corn             0         22           0          0           8              0              0          1           0          17
#  topic.crude            3          2         101          0           0              0              0         25           1           1
#  topic.earn            28          3           3       1022           0              0              1          2           0           1
#  topic.grain            0          0           0          0           6              0              0          0           0           0
#  topic.interest         0          2           0          0           1             77             18          0           0           0
#  topic.money.fx         0          0           1          0           1             18             67          1           5           1
#  topic.ship             1          1           7          0           4              0              5         44           0           3
#  topic.trade            1          3           1          0           4              4              2          2          97           4
#  topic.wheat            0         10           0          0           5              0              0          2           1          19



#evaluation part#
#+++++++++++++++#


#write two evaluation functions to save time

#this function for accuracy, precision and recall
evaluation<-function(table){
    rowEval<-c("topic.acq", "topic.corn", "topic.crude", "topic.earn", "topic.grain", "topic.interest", "topic.money.fx", "topic.ship", "topic.trade", "topic.wheat")
    colEval<-c("TN", "TP", "FN", "FP", "Accuracy", "Recall", "Precision", "F-measure")
    eval<-matrix(NA, 10, 8, dimnames = list(rowEval, colEval))
    
    for(i in 1:nrow(table)){
        TP <- table[i,i]
        FN <- sum(table[,i], na.rm = TRUE) - table[i,i]
        FP <- sum(table[i,], na.rm = TRUE) - table[i,i]
        TN <- sum(table)-TP-FN-FP
        
        Accuracy <- (TP + TN) / sum(table)
        Recall <- TP / (TP + FN)
        Precision <- TP / (TP + FP)
        Fmeasure <- 2 * Precision * Recall / (Precision + Recall)
        
        eval[i,1] <- TN
        eval[i,2] <- TP
        eval[i,3] <- FN
        eval[i,4] <- FP
        eval[i,5] <- Accuracy
        eval[i,6] <- Recall
        eval[i,7] <- Precision 
        eval[i,8] <- Fmeasure  
    }
    return(eval)
}


#this function for micro and macro averaged measures
#input is the return of evaluation() function
evaluationM<-function(eval){    
    rowEvalM<-c("Macro", "Micro")
    colEvalM<-c("Recall", "Precision")
    evalM<-matrix(NA, 2, 2, dimnames = list(rowEvalM, colEvalM)) 
    
    evalM[1,1] <- sum(eval[,6]) / 10
    evalM[1,2] <- sum(eval[,7]) / 10
    evalM[2,1] <- sum(eval[,2]) / sum(eval[,2]+eval[,3])
    evalM[2,2] <- sum(eval[,2]) / sum(eval[,2]+eval[,4])
       
    return(evalM)   
}



#SVM
evaluation(SVMLDATable)
#                 TN   TP FN  FP  Accuracy     Recall Precision  F-measure
#topic.acq      1604  601 34  79 0.9512511 0.94645669 0.8838235 0.91406844
#topic.corn     2253   19 29  17 0.9801553 0.39583333 0.5277778 0.45238095
#topic.crude    2172   82 43  21 0.9723900 0.65600000 0.7961165 0.71929825
#topic.earn     1175 1018 23 102 0.9460742 0.97790586 0.9089286 0.94215641
#topic.grain    2288    1 29   0 0.9874892 0.03333333 1.0000000 0.06451613
#topic.interest 2199   63 39  17 0.9758412 0.61764706 0.7875000 0.69230769
#topic.money.fx 2202   50 49  17 0.9715272 0.50505051 0.7462687 0.60240964
#topic.ship     2205   46 37  30 0.9710958 0.55421687 0.6052632 0.57861635
#topic.trade    2182   83 24  29 0.9771355 0.77570093 0.7410714 0.75799087
#topic.wheat    2251   24 24  19 0.9814495 0.50000000 0.5581395 0.52747253

mean(evaluation(SVMLDATable)[,8])
#[1] 0.6251217

evaluationM(evaluation(SVMLDATable))
#         Recall Precision
#Macro 0.5962145 0.7554889
#Micro 0.8572045 0.8572045



#naiveBayes
evaluation(NBLDATable)
#                 TN  TP  FN  FP  Accuracy    Recall  Precision  F-measure
#topic.acq      1652 436 199  31 0.9007765 0.6866142 0.93361884 0.79128857
#topic.corn     2254  11  37  16 0.9771355 0.2291667 0.40740741 0.29333333
#topic.crude    2130  33  92  63 0.9331320 0.2640000 0.34375000 0.29864253
#topic.earn     1266 925 116  11 0.9452114 0.8885687 0.98824786 0.93576125
#topic.grain    2143   4  26 145 0.9262295 0.1333333 0.02684564 0.04469274
#topic.interest 2162  57  45  54 0.9572908 0.5588235 0.51351351 0.53521127
#topic.money.fx 2183  29  70  36 0.9542709 0.2929293 0.44615385 0.35365854
#topic.ship     2189  16  67  46 0.9512511 0.1927711 0.25806452 0.22068966
#topic.trade    2190  34  73  21 0.9594478 0.3177570 0.61818182 0.41975309
#topic.wheat    1960  40   8 310 0.8628128 0.8333333 0.11428571 0.20100503

mean(evaluation(NBLDATable)[,8])
#[1] 0.4094036

evaluationM(evaluation(NBLDATable))
#         Recall Precision
#Macro 0.4397297 0.4650069
#Micro 0.6837791 0.6837791



#randomForest
evaluation(RFLDATable)
#                 TN   TP FN FP  Accuracy    Recall Precision F-measure
#topic.acq      1626  602 33 57 0.9611734 0.9480315 0.9135053 0.9304482
#topic.corn     2244   22 26 26 0.9775669 0.4583333 0.4583333 0.4583333
#topic.crude    2161  101 24 32 0.9758412 0.8080000 0.7593985 0.7829457
#topic.earn     1239 1022 19 38 0.9754098 0.9817483 0.9641509 0.9728701
#topic.grain    2288    6 24  0 0.9896462 0.2000000 1.0000000 0.3333333
#topic.interest 2195   77 25 21 0.9801553 0.7549020 0.7857143 0.7700000
#topic.money.fx 2192   67 32 27 0.9745470 0.6767677 0.7127660 0.6943005
#topic.ship     2214   44 39 21 0.9741156 0.5301205 0.6769231 0.5945946
#topic.trade    2190   97 10 21 0.9866264 0.9065421 0.8220339 0.8622222
#topic.wheat    2252   19 29 18 0.9797239 0.3958333 0.5135135 0.4470588

mean(evaluation(RFLDATable)[,8])
#[1] 0.6846107

evaluationM(evaluation(RFLDATable))
#         Recall Precision
#Macro 0.6660279 0.7606339
#Micro 0.8874029 0.8874029


#3.2 use TF*IDF as features#
#==========================#

reutersTFIDFf<-cbind(reutersTop,TFIDFf)
#add a column to represent topics
reutersTFIDFf[,"topic"]<-"empty"

# 1-purpose
# 2:11-Top10 topics
# 12:179-LDA features
# 180-topic

#use topicBackup
#strategy:assign only the minority class
reutersTFIDFf[,180]<-topicBackup

#tempNA
reutersTFIDFf<-reutersTFIDFf[-tempNA,]
#now has 8599 obs

#split Train and Test dataset
TFIDFTrain<-reutersTFIDFf[which(reutersTFIDFf$purpose=="train"),]
TFIDFTest<-reutersTFIDFf[which(reutersTFIDFf$purpose=="test"),]

#remove useless column
TFIDFTrain<-TFIDFTrain[,-c(1:11)]
TFIDFTest<-TFIDFTest[,-c(1:11)]

write.csv(TFIDFTrain, "TFIDFTrain.csv", row.names = F)
write.csv(TFIDFTest, "TFIDFTest.csv", row.names = F)



#begin to model and predict#
#++++++++++++++++++++++++++#


#SVM
SVMTFIDFTrain<-TFIDFTrain
SVMTFIDFTest<-TFIDFTest

SVMTFIDFModel <- svm(topic ~ .,SVMTFIDFTrain)
SVMTFIDFPre <- predict(SVMTFIDFModel,SVMTFIDFTest)
SVMTFIDFTable <- table(SVMTFIDFPre, SVMTFIDFTest$topic)

#SVMTFIDFPre      topic.acq topic.corn topic.crude topic.earn topic.grain topic.interest topic.money.fx topic.ship topic.trade topic.wheat
#  topic.acq            593          6          13         19           3              6             22         13           2           4
#  topic.corn             1         24           1          0           9              0              0          2           6          11
#  topic.crude            4          0          99          0           2              0              0         13           1           1
#  topic.earn            32          9           4       1019           4              7              5          1           7           6
#  topic.grain            0          0           0          0           3              0              0          0           0           0
#  topic.interest         0          0           1          1           0             71             14          0           0           0
#  topic.money.fx         2          0           0          1           1             13             53          2           5           0
#  topic.ship             0          2           5          0           2              0              1         45           1           6
#  topic.trade            3          2           2          1           2              5              4          5          84           4
#  topic.wheat            0          5           0          0           4              0              0          2           1          16



#naiveBayes
NBTFIDFTrain<-TFIDFTrain
NBTFIDFTest<-TFIDFTest

NBTFIDFModel <- naiveBayes(topic ~ .,NBTFIDFTrain)
NBTFIDFPre <- predict(NBTFIDFModel,NBTFIDFTest)
NBTFIDFTable <- table(NBTFIDFPre, NBTFIDFTest$topic)

#NBTFIDFPre       topic.acq topic.corn topic.crude topic.earn topic.grain topic.interest topic.money.fx topic.ship topic.trade topic.wheat
#  topic.acq            516          1           9         32           0              2              4          0           0           0
#  topic.corn             6         14           3          5           3              0              0          5          11           3
#  topic.crude           12          3          61         18           1              1              0          3           0           1
#  topic.earn            14          0           0        861           0              0              0          0           0           1
#  topic.grain            8          9           8          3           9              2              2          7           4           8
#  topic.interest        18          1           2         49           0             58             22          5           3           1
#  topic.money.fx         6          2           2          3           1             34             52          1           7           0
#  topic.ship            30          0          22          8           1              1              5         43           5           3
#  topic.trade            5          3           7          6           5              3             11          7          66           1
#  topic.wheat           20         15          11         56          10              1              3         12          11          30



#randomForest
RFTFIDFTrain<-TFIDFTrain
RFTFIDFTest<-TFIDFTest

RFTFIDFModel <- randomForest(topic ~ .,RFTFIDFTrain)
RFTFIDFPre <- predict(RFTFIDFModel, RFTFIDFTest)
RFTFIDFTable <- table(RFTFIDFPre, RFTFIDFTest$topic)

#RFTFIDFPre       topic.acq topic.corn topic.crude topic.earn topic.grain topic.interest topic.money.fx topic.ship topic.trade topic.wheat
#  topic.acq            605          3           9         19           1              4              5          7           2           2
#  topic.corn             0         29           0          0          12              0              0          1           0          12
#  topic.crude            5          1         108          0           1              0              0         25           1           1
#  topic.earn            24          1           2       1022           0              0              0          2           0           0
#  topic.grain            0          0           0          0           2              0              0          0           0           1
#  topic.interest         0          1           0          0           1             75             19          0           0           0
#  topic.money.fx         0          0           0          0           1             21             66          2           6           0
#  topic.ship             0          4           5          0           2              0              6         40           0           6
#  topic.trade            1          2           1          0           4              2              3          4          97           4
#  topic.wheat            0          7           0          0           6              0              0          2           1          22



#evaluation part#
#+++++++++++++++#


#SVM
evaluation(SVMTFIDFTable)
#                 TN   TP FN FP  Accuracy    Recall Precision F-measure
#topic.acq      1595  593 42 88 0.9439172 0.9338583 0.8707783 0.9012158
#topic.corn     2240   24 24 30 0.9767041 0.5000000 0.4444444 0.4705882
#topic.crude    2172   99 26 21 0.9797239 0.7920000 0.8250000 0.8081633
#topic.earn     1202 1019 22 75 0.9581536 0.9788665 0.9314442 0.9545667
#topic.grain    2288    3 27  0 0.9883520 0.1000000 1.0000000 0.1818182
#topic.interest 2200   71 31 16 0.9797239 0.6960784 0.8160920 0.7513228
#topic.money.fx 2195   53 46 24 0.9698016 0.5353535 0.6883117 0.6022727
#topic.ship     2218   45 38 17 0.9762726 0.5421687 0.7258065 0.6206897
#topic.trade    2183   84 23 28 0.9779983 0.7850467 0.7500000 0.7671233
#topic.wheat    2258   16 32 12 0.9810181 0.3333333 0.5714286 0.4210526

mean(evaluation(SVMTFIDFTable)[,8])
#[1] 0.6478813

evaluationM(evaluation(SVMTFIDFTable))
#         Recall Precision
#Macro 0.6196705 0.7623306
#Micro 0.8658326 0.8658326



#naiveBayes
evaluation(NBTFIDFTable)
#                 TN  TP  FN  FP  Accuracy    Recall Precision F-measure
#topic.acq      1635 516 119  48 0.9279551 0.8125984 0.9148936 0.8607173
#topic.corn     2234  14  34  36 0.9698016 0.2916667 0.2800000 0.2857143
#topic.crude    2154  61  64  39 0.9555651 0.4880000 0.6100000 0.5422222
#topic.earn     1262 861 180  15 0.9158758 0.8270893 0.9828767 0.8982786
#topic.grain    2237   9  21  51 0.9689387 0.3000000 0.1500000 0.2000000
#topic.interest 2115  58  44 101 0.9374461 0.5686275 0.3647799 0.4444444
#topic.money.fx 2163  52  47  56 0.9555651 0.5252525 0.4814815 0.5024155
#topic.ship     2160  43  40  75 0.9503883 0.5180723 0.3644068 0.4278607
#topic.trade    2163  66  41  48 0.9616048 0.6168224 0.5789474 0.5972851
#topic.wheat    2131  30  18 139 0.9322692 0.6250000 0.1775148 0.2764977

mean(evaluation(NBTFIDFTable)[,8])
#[1] 0.5035436

evaluationM(evaluation(NBTFIDFTable))
#         Recall Precision
#Macro 0.5573129 0.4904901
#Micro 0.7377049 0.7377049



#randomForest
evaluation(RFTFIDFTable)
#                 TN   TP FN FP  Accuracy     Recall Precision F-measure
#topic.acq      1631  605 30 52 0.9646247 0.95275591 0.9208524 0.9365325
#topic.corn     2245   29 19 25 0.9810181 0.60416667 0.5370370 0.5686275
#topic.crude    2159  108 17 34 0.9779983 0.86400000 0.7605634 0.8089888
#topic.earn     1248 1022 19 29 0.9792925 0.98174832 0.9724072 0.9770554
#topic.grain    2287    2 28  1 0.9874892 0.06666667 0.6666667 0.1212121
#topic.interest 2195   75 27 21 0.9792925 0.73529412 0.7812500 0.7575758
#topic.money.fx 2189   66 33 30 0.9728214 0.66666667 0.6875000 0.6769231
#topic.ship     2212   40 43 23 0.9715272 0.48192771 0.6349206 0.5479452
#topic.trade    2190   97 10 21 0.9866264 0.90654206 0.8220339 0.8622222
#topic.wheat    2254   22 26 16 0.9818809 0.45833333 0.5789474 0.5116279

mean(evaluation(RFTFIDFTable)[,8])
#[1] 0.676871

evaluationM(evaluation(RFTFIDFTable))
#         Recall Precision
#Macro 0.6718101 0.7362179
#Micro 0.8912856 0.8912856



#3.3 use LDA+TF*IDF as features#
#==============================#


reutersLDATFIDFf<-cbind(reutersTop,LDATDMf,TFIDFf)
#add a column to represent topics
reutersLDATFIDFf[,"topic"]<-"empty"

# 1-purpose
# 2:11-Top10 topics
# 12:179-LDA features
# 180:347-TF*IDF features
# 348-topic

#use topicBackup
#strategy:assign only the minority class
reutersLDATFIDFf[,348]<-topicBackup

#tempNA
reutersLDATFIDFf<-reutersLDATFIDFf[-tempNA,]
#now has 8599 obs

#split Train and Test dataset
LDATFIDFTrain<-reutersLDATFIDFf[which(reutersLDATFIDFf$purpose=="train"),]
LDATFIDFTest<-reutersLDATFIDFf[which(reutersLDATFIDFf$purpose=="test"),]

#remove useless column
LDATFIDFTrain<-LDATFIDFTrain[,-c(1:11)]
LDATFIDFTest<-LDATFIDFTest[,-c(1:11)]

write.csv(LDATFIDFTrain, "LDATFIDFTrain.csv", row.names = F)
write.csv(LDATFIDFTest, "LDATFIDFTest.csv", row.names = F)


#begin to model and predict#
#++++++++++++++++++++++++++#


#SVM
SVMLDATFIDFTrain<-LDATFIDFTrain
SVMLDATFIDFTest<-LDATFIDFTest

SVMLDATFIDFModel <- svm(topic ~ .,SVMLDATFIDFTrain)
SVMLDATFIDFPre <- predict(SVMLDATFIDFModel,SVMLDATFIDFTest)
SVMLDATFIDFTable <- table(SVMLDATFIDFPre, SVMLDATFIDFTest$topic)

#SVMLDATFIDFPre   topic.acq topic.corn topic.crude topic.earn topic.grain topic.interest topic.money.fx topic.ship topic.trade topic.wheat
#  topic.acq            603          3          13         19           2              4              9          5           1           0
#  topic.corn             1         23           1          0           7              0              0          2           5          12
#  topic.crude            1          1          94          0           3              0              1         11           0           1
#  topic.earn            27          8           4       1020           3              8              6          3           7           5
#  topic.grain            0          0           0          0           3              0              0          0           0           0
#  topic.interest         0          0           1          1           0             72             13          0           0           0
#  topic.money.fx         1          0           1          1           1             12             62          1           5           0
#  topic.ship             1          2           7          0           2              0              1         53           2           7
#  topic.trade            1          4           4          0           5              6              7          6          87           4
#  topic.wheat            0          7           0          0           4              0              0          2           0          19



#naiveBayes
NBLDATFIDFTrain<-LDATFIDFTrain
NBLDATFIDFTest<-LDATFIDFTest

NBLDATFIDFModel <- naiveBayes(topic ~ .,NBLDATFIDFTrain)
NBLDATFIDFPre <- predict(NBLDATFIDFModel,NBLDATFIDFTest)
NBLDATFIDFTable <- table(NBLDATFIDFPre, NBLDATFIDFTest$topic)

#NBLDATFIDFPre    topic.acq topic.corn topic.crude topic.earn topic.grain topic.interest topic.money.fx topic.ship topic.trade topic.wheat
#  topic.acq            538          1          12         27           0              1              1          0           0           0
#  topic.corn             1         11           3          2           2              0              0          3           5           1
#  topic.crude           20          3          57         26           0              0              0          3           1           1
#  topic.earn            10          0           0        948           0              0              0          0           0           1
#  topic.grain           12          8          12          3           9              2              3         10           7           6
#  topic.interest         7          1           1          6           1             63             28          2           3           0
#  topic.money.fx         4          2           2          0           0             28             50          1           9           0
#  topic.ship            16          1          20          1           1              2              6         34           6           4
#  topic.trade            3          3           5          3           4              4              8          4          57           0
#  topic.wheat           24         18          13         25          13              2              3         26          19          35



#randomForest
RFLDATFIDFTrain<-LDATFIDFTrain
RFLDATFIDFTest<-LDATFIDFTest

RFLDATFIDFModel <- randomForest(topic ~ .,RFLDATFIDFTrain)
RFLDATFIDFPre <- predict(RFLDATFIDFModel, RFLDATFIDFTest)
RFLDATFIDFTable <- table(RFLDATFIDFPre, RFLDATFIDFTest$topic)

#RFLDATFIDFPre    topic.acq topic.corn topic.crude topic.earn topic.grain topic.interest topic.money.fx topic.ship topic.trade topic.wheat
#  topic.acq            597          4           9         20           0              4              3          4           3           2
#  topic.corn             0         27           1          0          13              0              0          3           0          14
#  topic.crude            6          1         106          0           0              0              0         26           1           1
#  topic.earn            30          1           2       1021           0              0              0          2           0           0
#  topic.grain            0          0           0          0           2              0              0          0           0           1
#  topic.interest         0          2           0          0           1             75             19          0           0           0
#  topic.money.fx         0          0           0          0           1             21             67          1           5           0
#  topic.ship             0          3           6          0           4              0              7         43           0           5
#  topic.trade            2          3           1          0           4              2              3          2          96           4
#  topic.wheat            0          7           0          0           5              0              0          2           2          21



#evaluation part#
#+++++++++++++++#


#SVM
evaluation(SVMLDATFIDFTable)
#                 TN   TP FN FP  Accuracy    Recall Precision F-measure
#topic.acq      1627  603 32 56 0.9620362 0.9496063 0.9150228 0.9319938
#topic.corn     2242   23 25 28 0.9771355 0.4791667 0.4509804 0.4646465
#topic.crude    2175   94 31 18 0.9788611 0.7520000 0.8392857 0.7932489
#topic.earn     1206 1020 21 71 0.9603106 0.9798271 0.9349221 0.9568480
#topic.grain    2288    3 27  0 0.9883520 0.1000000 1.0000000 0.1818182
#topic.interest 2201   72 30 15 0.9805867 0.7058824 0.8275862 0.7619048
#topic.money.fx 2197   62 37 22 0.9745470 0.6262626 0.7380952 0.6775956
#topic.ship     2213   53 30 22 0.9775669 0.6385542 0.7066667 0.6708861
#topic.trade    2174   87 20 37 0.9754098 0.8130841 0.7016129 0.7532468
#topic.wheat    2257   19 29 13 0.9818809 0.3958333 0.5937500 0.4750000

mean(evaluation(SVMLDATFIDFTable)[,8])
#[1] 0.6667189

evaluationM(evaluation(SVMLDATFIDFTable))
#         Recall Precision
#Macro 0.6440217 0.7707922
#Micro 0.8783434 0.8783434



#naiveBayes
evaluation(NBLDATFIDFTable)
#                 TN  TP FN  FP  Accuracy    Recall Precision F-measure
#topic.acq      1641 538 97  42 0.9400345 0.8472441 0.9275862 0.8855967
#topic.corn     2253  11 37  17 0.9767041 0.2291667 0.3928571 0.2894737
#topic.crude    2139  57 68  54 0.9473684 0.4560000 0.5135135 0.4830508
#topic.earn     1266 948 93  11 0.9551337 0.9106628 0.9885297 0.9480000
#topic.grain    2225   9 21  63 0.9637619 0.3000000 0.1250000 0.1764706
#topic.interest 2167  63 39  49 0.9620362 0.6176471 0.5625000 0.5887850
#topic.money.fx 2173  50 49  46 0.9590164 0.5050505 0.5208333 0.5128205
#topic.ship     2178  34 49  57 0.9542709 0.4096386 0.3736264 0.3908046
#topic.trade    2177  57 50  34 0.9637619 0.5327103 0.6263736 0.5757576
#topic.wheat    2127  35 13 143 0.9327006 0.7291667 0.1966292 0.3097345

mean(evaluation(NBLDATFIDFTable)[,8])
#[1] 0.5160494

evaluationM(evaluation(NBLDATFIDFTable))
#         Recall Precision
#Macro 0.5537287 0.5227449
#Micro 0.7773943 0.7773943



#randomForest
evaluation(RFLDATFIDFTable)
#                 TN   TP FN FP  Accuracy     Recall Precision F-measure
#topic.acq      1634  597 38 49 0.9624676 0.94015748 0.9241486 0.9320843
#topic.corn     2239   27 21 31 0.9775669 0.56250000 0.4655172 0.5094340
#topic.crude    2158  106 19 35 0.9767041 0.84800000 0.7517730 0.7969925
#topic.earn     1242 1021 20 35 0.9762726 0.98078770 0.9668561 0.9737721
#topic.grain    2287    2 28  1 0.9874892 0.06666667 0.6666667 0.1212121
#topic.interest 2194   75 27 22 0.9788611 0.73529412 0.7731959 0.7537688
#topic.money.fx 2191   67 32 28 0.9741156 0.67676768 0.7052632 0.6907216
#topic.ship     2210   43 40 25 0.9719586 0.51807229 0.6323529 0.5695364
#topic.trade    2190   96 11 21 0.9861950 0.89719626 0.8205128 0.8571429
#topic.wheat    2254   21 27 16 0.9814495 0.43750000 0.5675676 0.4941176

mean(evaluation(RFLDATFIDFTable)[,8])
#[1] 0.6698782

evaluationM(evaluation(RFLDATFIDFTable))
#         Recall Precision
#Macro 0.6662942 0.7273854
#Micro 0.8865401 0.8865401



#based on the evaluation above, LDA+TF*IDF performs best
#so we will use this features to do the cluster



#==========#
#  Task 4  #
#==========#


#Clustering#
#==========#


#based on the evaluation in task 3, LDA+TF*IDF performs best
#so we will use this features to do the cluster
LDATDMf<-read.csv(file="LDATDMf.csv",header=T,sep=",")
TFIDFf<-read.csv(file="TFIDFf.csv",header=T,sep=",")

clusterLDATFIDF<-cbind(LDATDMf,TFIDFf)
d <- dist(clusterLDATFIDF)



#begin to cluster#
#++++++++++++++++#

#K-means
clusterKM <- kmeans(clusterLDATFIDF, 10)


#Hierarchical Agglomerative
clusterH <- hclust(d)
clusterHA <- cutree(clusterH, k = 10)


#DBSCAN
library(fpc)
clusterD<-dbscan(clusterLDATFIDF,4,MinPts=5,method="hybrid")


#PAM
#library(cluster)
#clusterPAM <- pam(clusterLDATFIDF, 10)



#evaluation part#
#+++++++++++++++#

#K-means
silKM <- silhouette(clusterKM$cluster, d)
mean(silKM[,3])
#[1] 0.03797562


#Hierarchical Agglomerative
silHA <- silhouette(clusterHA, d)
mean(silHA[,3])
#[1] 0.3826627
#so this performs best!


#DBSCAN
silD <- silhouette(clusterD$cluster, d)
mean(silD[,3])
#[1] -0.08132438


#PAM
#silPAM <- silhouette(clusterPAM$cluster, d)
#mean(silPAM[,3])
#[1] -0.04659366


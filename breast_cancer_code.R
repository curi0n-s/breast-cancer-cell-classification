# 
# PSY 810 Project
# June 9 2020


# Classification of Breast Cancer using digitized images of fine needle aspirates of breast mass
# Dataset from: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

library(tidyverse)     
library(ggplot2)      
library(corrplot)
library(caret)
library(corrr)
library(kernlab)  
library(e1071)    
library(DT)
library(nnet)
#require(ggiraph)
require(plyr)
#library(ggiraphExtra)
library(corrplot)insta
library(data.table)
#library(ggpubr)
#library(kableExtra)
library(pwr)
library(rms)
library(leaps)
library(bestglm)
library(resample)
library(graphics)
library(pracma)
library(boot)
library(ISLR)
library(MASS)
library(dplyr)
library(randomForest)

#_____________________________________________________________________________________________
# 1. Import Dataset, Sample Plots, Create Test and Training Data
#_____________________________________________________________________________________________

rawData = read.csv("wdbc.csv", header=TRUE, na.strings = c("",NA))
morphData = data.table(rawData[,2:ncol(rawData)])

ggplot(data = morphData, aes(x = Diagnosis, fill = Diagnosis)) +
  geom_bar()+
  labs(x = "Classification",
      y = "Count",
      title = "357 Benign Samples, 212 Malignant")

morphData %>%
  ggplot(aes(x = mean_area,
             color = Diagnosis,
             fill = Diagnosis)) +
  geom_density(alpha=.25) +
  theme_bw() +
  labs(x = "Area Mean",
       y = "Density",
       title = "Malignant Cell Nuclei Have Higher Area")

morphData %>%
  ggplot(aes(x = mean_num_cc_pts,
             color = Diagnosis,
             fill = Diagnosis)) +
  geom_density(alpha=.25) +
  theme_bw() +
  labs(x = "Area Mean",
       y = "Density",
       title = "Malignant Cell Nuclei Have More Concave Points")

morphData %>%
  ggplot(aes(x = mean_perimeter,
             color = Diagnosis,
             fill = Diagnosis)) +
  geom_density(alpha=.25) +
  theme_bw() +
  labs(x = "Area Mean",
       y = "Density",
       title = "Malignant Cell Nuclei Have Larger Perimeters")

# how to LDA and QDA look?
morphData %>% gather() %>% head()
ggplot(gather(morphData[,2:ncol(morphData)]), aes(value)) + 
  geom_histogram(bins = 30) + 
  facet_wrap(~key, scales = 'free_x')

colVariance=colVars(morphData[,2:ncol(morphData)])
min(colVariance)
max(colVariance)
mean(colVariance)
sd(colVariance) #....no shot for LDA

#are decision boundaries between classes linear? if so QDA will likely be outperformed by Logistic Regression (per PSY810 lecture 3 slide 51)

panel.hist <- function(x, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5) )
  h <- hist(x, plot = FALSE)
  breaks <- h$breaks; nB <- length(breaks)
  y <- h$counts; y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col = "cyan", ...)
}
colors <- c('salmon', 'cyan')[unclass(morphData$Diagnosis)]

pairs(morphData[,2:5], 
      cex = 1.5, pch = 21, bg = colors, horOdd=TRUE,
      diag.panel = panel.hist, cex.labels = 1.5, font.labels = 1.5)
pairs(morphData[,6:10],
      cex = 1.5, pch = 21, bg = colors, horOdd=TRUE,
      diag.panel = panel.hist, cex.labels = 1, font.labels = 1.5)
pairs(morphData[,11:20],
      cex = 1.5, pch = 21, bg = colors, horOdd=TRUE,
      diag.panel = panel.hist, cex.labels = 1, font.labels = 1)
pairs(morphData[,21:30],
      cex = 1.5, pch = 21, bg = colors, horOdd=TRUE,
      diag.panel = panel.hist, cex.labels = 1, font.labels = 1)

# it seems most data that has any sort of clear seperability is linearly seperable, 
# ie no "bullseye" type boundaries that require nonlinear segmentation

colnames(morphData)
morphData$Diagnosis = as.character(morphData$Diagnosis)
#change M/B to 1/0 for prediction accuracy calculation
morphData$Diagnosis = gsub("M",1,morphData$Diagnosis)
morphData$Diagnosis = gsub("B",0,morphData$Diagnosis)
morphData$Diagnosis = as.integer(morphData$Diagnosis)
sum(is.na(morphData)) # any nans?

#morphData = as.factor(morphData)
#as.factor(morphData$Diagnosis)


#train and test sets
#stratified sampling maintains equal representation of M/B in test and train
indices.train = createDataPartition(morphData$Diagnosis, p=0.8, list=FALSE)
morphData.train = morphData[indices.train,]
morphData.test = morphData[-indices.train,]

#_____________________________________________________________________________________________
# 2. Logistic Regression Model for all predictors (30 Predictors)
#_____________________________________________________________________________________________
#first all predictors, because statistical power of 30 predictors over 569 obs. is good
#this is becuase: 

#check statistical power
a=0.05
p=30
r2=0.1 
f2 = r2/(1-r2) #cohen effect size
#power to detect 95% of differences where they exist at p<0.05
statPower = pwr.f2.test(u=p,f2 = f2,sig.level=0.05,power=0.95)
minSampleSize = ceiling(statPower$v)+statPower$u+1
minSampleSize #is well over

#plus Peduzzi rule of Instances/Predictors = 15+

#training
set.seed(1)

logReg.fit = glm(Diagnosis~.,data=morphData.train,family=binomial,control = list(maxit = 50))
logReg.summary = summary(logReg.fit)
logReg.probs = predict(logReg.fit, type="response")
positiveThreshold = 0.5 #minimum probability for which positive diagnosis can be rendered ie 0.51 is considered probable malignancy
logReg.preds = ifelse(logReg.probs > positiveThreshold,1,0)
table(logReg.preds,morphData.train$Diagnosis)
acc_train1=mean(logReg.preds == morphData.train$Diagnosis)
print("TRAIN CLASSIFICATION RATE ALL IND VARS")
acc_train1
#testing
logReg.testProbs = predict(logReg.fit, newdata = morphData.test, type="response")
logReg.testPreds = ifelse(logReg.testProbs > positiveThreshold,1,0)
table(logReg.testPreds, morphData.test$Diagnosis)
acc_test1=mean(logReg.testPreds == morphData.test$Diagnosis) #mean of vector where ==1 if pred=truth, ==0 if not. take mean to find accuracy
print("TEST CLASSIFICATION RATE ALL IND VARS")
acc_test1

set.seed(1)
#which variables have highest changes in explanatory power? saw this for quasibinomial glm, but says innapropriate for binomial
remove2 <- update(logReg.fit, ~. - morphData.train[,3])
anova(logReg.fit,remove2,test="F")
for (j in 1:ncol(morphData[,2:ncol(morphData)])){
  a=1
}
#improvements with different variables?

#significance function from http://www.sthda.com/english/wiki/visualize-correlation-matrix-using-correlogram

# which variables are most correlated?
#significance function
# mat : is a matrix of data
# ... : further arguments to pass to the native R cor.test function
cor.mtest <- function(mat, ...) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat<- matrix(NA, n, n)
  diag(p.mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], ...)
      p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
    }
  }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}

corrdata = morphData[,-c(1,2)]
p.mat = cor.mtest(corrdata)
corrplot(cor(corrdata), method="circle",order = 'hclust',p.mat = p.mat, sig.level = 0.01,diag=TRUE,insig="pch")

highly_correlated = findCorrelation(cor(corrdata), cutoff = 0.9)
p.mat = cor.mtest(corrdata[, c(6,  7, 22, 20,  2, 23, 12, 13,  1)])
corrplot(cor(corrdata[, c(6,  7, 22, 20,  2, 23, 12, 13,  1)]),method="number", order = "hclust", p.mat = p.mat)

#remove all with higher rsq than 0.9: mean_num_cc_pts, max_area, max_perimeter, max_radius, std_error_area
morphDataSmall = dplyr::select(morphData, - mean_num_cc_pts, -max_area, -max_perimeter, -max_radius, -std_error_area)

#second logistic regression
#train and test sets
#stratified sampling maintains equal representation of M/B in test and train
indices.train = createDataPartition(morphDataSmall$Diagnosis, p=0.8, list=FALSE)
morphDataSmall.train = morphDataSmall[indices.train,]
morphDataSmall.test = morphDataSmall[-indices.train,]

#training

logReg2.fit = glm(Diagnosis~.,data=morphDataSmall.train,family=binomial,control = list(maxit = 50))
logReg2.summary = summary(logReg2.fit)
logReg2.probs = predict(logReg2.fit, type="response")
positiveThreshold = 0.5 #minimum probability for which positive diagnosis can be rendered ie 0.51 is considered probable malignancy
logReg2.preds = ifelse(logReg2.probs > positiveThreshold,1,0)
table(logReg2.preds,morphDataSmall.train$Diagnosis)
acc_train2=mean(logReg2.preds == morphDataSmall.train$Diagnosis)
print("TRAIN CLASSIFICATION RATE, SANS COLINEAR INDEPENDENT VARIABLES")
acc_train2
#testing
logReg2.testProbs = predict(logReg2.fit, newdata = morphDataSmall.test, type="response")
logReg2.testPreds = ifelse(logReg2.testProbs > positiveThreshold,1,0)
table(logReg2.testPreds, morphDataSmall.test$Diagnosis)
acc_test2=mean(logReg2.testPreds == morphDataSmall.test$Diagnosis) #mean of vector where ==1 if pred=truth, ==0 if not. take mean to find accuracy
print("TEST CLASSIFICATION RATE, SANS COLINEAR INDEPENDENT VARIABLES")
acc_test2

tabLR = table(logReg2.preds,morphDataSmall.train$Diagnosis)
specificity(tabLR)
sensitivity(tabLR)

#k-fold (k=10) validation to compare distribution of CLASSIFICATION RATE across dataset
#tuning of threshold parameter for best test error
k=10
allInds = 1:nrow(morphDataSmall)
kinds = seq(1,nrow(morphDataSmall),k)
notKinds = allInds
acc_train = matrix(,10,100)
acc_test = matrix(,10,100)
k10.trainError = rep(0,10)
k10.testError = rep(0,10)
kData<-morphDataSmall[sample(nrow(morphDataSmall)),] #shuffle data
positiveThreshold = 1 #minimum probability for which positive diagnosis can be rendered ie 0.51 is considered probable malignancy

for(i in 1:10){
  set.seed(1)
  thisTest = kData[kinds[i]:kinds[i+1],]
  thisTrain = kData[allInds[-(kinds[i]:kinds[i+1])],]
  k10.glm.fit = glm(Diagnosis~.,data=thisTrain,family=binomial,control = list(maxit = 50))
  k10.probs = predict(k10.glm.fit, type="response")
  
  for(j in 1:100){
  set.seed(1)
  positiveThreshold=(0.01*j)
  k10.preds = ifelse(k10.probs > positiveThreshold,1,0)
  acc_train[i,j]=mean(k10.preds == thisTrain$Diagnosis)
  k10.testProbs = predict(k10.glm.fit, newdata = thisTest, type="response")
  k10.testPreds = ifelse(k10.testProbs > positiveThreshold,1,0)
  acc_test[i,j]=mean(k10.testPreds == thisTest$Diagnosis) #mean of vector where ==1 if pred=truth, ==0 if not. take mean to find accuracy
  k10.trainError[j] = mean(acc_train[,j])
  k10.testError[j]=mean(acc_test[,j])
}
}

max_test_accuracy = max(k10.testError)
max_test_ind = which(k10.testError==max_test_accuracy)[1]
best_threshold = max_test_ind*0.01 #0.77 for set.seed(1)

test_data <-
  data.frame(
    train_error = k10.trainError,
    test_error = k10.testError,
    thresh = 0.01*(1:100), length.out=100)
ggplot(test_data, aes(thresh)) + 
  geom_line(aes(y = train_error,size=1, colour = "Train Accuracy")) + 
  geom_line(aes(y = test_error,size=1, colour = "Test Accuracy"))


#compare train and test acc over all phases
trainacc = c(acc_train1,acc_train2,k10.trainError[50], max(k10.trainError)[1])
testacc = c(acc_test1, acc_test2, k10.testError[50], max(k10.testError)[1])
test_data2 <-
  data.frame(
    train_error = trainacc,
    test_error = testacc,
    approach = 1:4)
ggplot(test_data2, aes(approach)) + 
  geom_line(aes(y = train_error,size=1, colour = "Train Accuracy")) + 
  geom_line(aes(y = test_error,size=1, colour = "Test Accuracy"))

print("MAX TEST ACCURACY AFTER PROCESSING")
max_test_accuracy

#what are probabilities of cancer?
k10.testProbs # mainly near 0 and 1
logReg2.summary #which are significant?
#_____________________________________________________________________________________________
#improvements with other techniques?
#_____________________________________________________________________________________________
#QDA

k=10
morphDataSmall$Diagnosis = gsub(1,2,morphDataSmall$Diagnosis) #QDA PREDICTS 2/1 as classes
morphDataSmall$Diagnosis = gsub(0,1,morphDataSmall$Diagnosis)

indices.train = createDataPartition(morphDataSmall$Diagnosis, p=0.8, list=FALSE)
morphDataSmall.train = morphDataSmall[indices.train,]
morphDataSmall.test = morphDataSmall[-indices.train,]

allInds = 1:nrow(morphDataSmall)
kinds = seq(1,nrow(morphDataSmall),k)
notKinds = allInds
kData<-morphDataSmall[sample(nrow(morphDataSmall)),] #shuffle data
positiveThreshold = 1

#w/o kfold

qda.fit=qda(Diagnosis~.,data=morphDataSmall.train)
qda.class=as.numeric(predict(qda.fit,morphDataSmall.test)$class)
mean(qda.class==morphDataSmall.test$Diagnosis)

#w 10-fold
qda.accuracy = rep(0,10)
qda.trainacc = rep(0,10)

for(i in 1:10){
  set.seed(1)
  thisTest = kData[kinds[i]:kinds[i+1],]
  thisTrain = kData[allInds[-(kinds[i]:kinds[i+1])],]
  qda.fit=qda(Diagnosis~.,data=thisTrain)
  qda.classt=as.numeric(predict(qda.fit,thisTrain)$class)
  qda.trainacc[i]=mean(qda.classt==thisTrain$Diagnosis)
  qda.class=as.numeric(predict(qda.fit,thisTest)$class)
  qda.accuracy[i]=mean(qda.class==thisTest$Diagnosis)
}
qda.10fold.trainacc = mean(qda.trainacc)
qda.10fold.testacc = mean(qda.accuracy)

#Linear SVM
lsvm.trainacc = rep(0,10)
lsvm.accuracy = rep(0,10)
for(i in 1:10){
  set.seed(1)
  thisTest = kData[kinds[i]:kinds[i+1],]
  thisTrain = kData[allInds[-(kinds[i]:kinds[i+1])],]
  lsvm.fit=svm(Diagnosis~., data=thisTrain, type='C-classification', kernel='linear')
  lsvm.classt=as.numeric(predict(lsvm.fit,thisTrain))
  lsvm.trainacc[i] = mean(lsvm.classt==as.numeric(thisTrain$Diagnosis))
  lsvm.class=predict(lsvm.fit,thisTest)
  lsvm.accuracy[i]=mean(lsvm.class==as.numeric(thisTest$Diagnosis))
}
lsvm.trainacc = mean(lsvm.trainacc)
lsvm.testacc = mean(lsvm.accuracy)

tune.out=tune(svm ,as.numeric(Diagnosis)~.,data=morphDataSmall ,kernel ="linear", 
              ranges =list(gamma = 10^(-5:-1), cost = 10^(-3:1)))
lsvm.best.cost=tune.out$best.parameters$cost
lsvm.best.gamma=tune.out$best.parameters$gamma
lsvm.best.error = tune.out$best.performance

#Radial SVM
rsvm.trainacc = rep(0,10)
rsvm.accuracy = rep(0,10)
for(i in 1:10){
  set.seed(1)
  thisTest = kData[kinds[i]:kinds[i+1],]
  thisTrain = kData[allInds[-(kinds[i]:kinds[i+1])],]
  rsvm.fit=svm(Diagnosis~., data=thisTrain, type='C-classification', kernel='radial',cost =1, gamma=0.1)
  rsvm.classt=as.numeric(predict(rsvm.fit,thisTrain))
  rsvm.class=predict(rsvm.fit,thisTest)
  rsvm.trainacc[i] = mean(rsvm.classt==as.numeric(thisTrain$Diagnosis))
  rsvm.accuracy[i]=mean(rsvm.class==as.numeric(thisTest$Diagnosis))
}
rsvm.trainacc = mean(rsvm.trainacc)
rsvm.testacc = mean(rsvm.accuracy)

tune.out=tune(svm ,as.numeric(Diagnosis)~.,data=morphDataSmall ,kernel ="radial", 
              ranges =list(gamma = 10^(-5:-1), cost = 10^(-3:1)))
rsvm.best.cost=tune.out$best.parameters$cost
rsvm.best.gamma=tune.out$best.parameters$gamma
rsvm.best.error = tune.out$best.performance

#Random Forest Competitive With LR (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2264-5)
morphDataSmall$Diagnosis = gsub(1,0,morphDataSmall$Diagnosis) #RF binary operator?
morphDataSmall$Diagnosis = gsub(2,1,morphDataSmall$Diagnosis)
rf.trainacc = matrix(,10,100)
rf.accuracy = matrix(,10,100)
rf.MEANtrainacc = rep(0,10)
rf.MEANtestacc = rep(0,10)
for(i in 1:10){
  for(j in 1:100){
  set.seed(1)
  thisTest = kData[kinds[i]:kinds[i+1],]
  thisTrain = kData[allInds[-(kinds[i]:kinds[i+1])],]
  rf.model <- randomForest(as.factor(Diagnosis) ~ ., data=thisTrain, ntree=j,proximity=T)
  rf.classt=as.numeric(predict(rf.model,thisTrain))
  rf.class=predict(rf.model,thisTest)
  rf.trainacc[i,j] = mean(rf.classt==as.numeric(thisTrain$Diagnosis))
  rf.accuracy[i,j]=mean(rf.class==as.numeric(thisTest$Diagnosis))
  rf.MEANtrainacc[j] = mean(rf.trainacc[,j])
  rf.MEANtestacc[j]=mean(rf.accuracy[,j])
  }
}

rf.max_test_accuracy = max(rf.MEANtestacc)
rf.max_test_ind = which(rf.MEANtestacc==rf.max_test_accuracy)[1]
rf.best_treenum = rf.max_test_ind 

test_data3 <-
  data.frame(
    train_error = rf.MEANtrainacc,
    test_error = rf.MEANtestacc,
    num_trees = 1:100, length.out=100)
ggplot(test_data3, aes(num_trees)) + 
  geom_line(aes(y = train_error,size=1, colour = "Train Accuracy")) + 
  geom_line(aes(y = test_error,size=1, colour = "Test Accuracy"))


#best result
trainacc2 = c(max(k10.trainError)[1],qda.10fold.trainacc,lsvm.trainacc,rsvm.trainacc[1],rf.MEANtrainacc[rf.max_test_ind])
testacc2 = c(max_test_accuracy,qda.10fold.testacc,(1-lsvm.best.error),(1-rsvm.best.error),rf.max_test_accuracy)
test_data2 <-
  data.frame(
    train_error2 = trainacc2,
    test_error2 = testacc2,
    approach = 1:5)
ggplot(test_data2, aes(approach)) + 
  geom_line(aes(y = train_error2,size=1, colour = "Train Accuracy")) + 
  geom_line(aes(y = test_error2,size=1, colour = "Test Accuracy"))

print("MAX TEST ACCURACY AFTER PROCESSING")
print("TIE BETWEEN LR AND LINEAR SVM")
max_test_accuracy
lsvm.testacc

#_____________________________________________________________________________________________
#Genetic Correlates based on ESR1 levels (https://data.world/deviramanan2016/nki-breast-cancer-data)
#_____________________________________________________________________________________________
rawGeneData = read.csv("NKI_cleaned.csv", header=TRUE, na.strings = c("",NA))
geneData = rawGeneData[,17:ncol(rawGeneData)]

#there are many genes here, so we will need to figure out which are most correlated to esr1 which will be the therapeutic target here

#write correlations to esr1, starting with a multiple linear regression, under the assumption that esr1 will not follow an on/off trend necessitating log reg
esr1SortedGeneData = geneData[order(geneData$esr1),]
gadphSortedGeneData = geneData[order(geneData$G3PDH_570)]

write.csv(cor(esr1SortedGeneData),file="esrSortedCorr.csv")
esr1CorrData = read.csv("esrSortedCorr.csv", header=TRUE, na.strings = c("",NA))
esr1Corr_toOthers = as.numeric(esr1CorrData[3:nrow(esr1CorrData),2])

maxCorrInd = which.max(esr1Corr_toOthers[esr1Corr_toOthers<1])
maxCorrRsq = max(esr1Corr_toOthers[esr1Corr_toOthers<1])

#sort for top correlates to esr1 with cor() function
#cor by default uses pearson which uses which type of error estimate (ie least squares?)

sortedRsq = sort.int(esr1Corr_toOthers[esr1Corr_toOthers<1], partial = NULL, na.last = NA, decreasing = TRUE,
                     method = c("auto", "shell", "quick", "radix"), index.return = TRUE)
geneNamesByDescCorr = colnames(esr1CorrData[1, sortedRsq$ix])
allRsqNames = data.frame(table(sortedRsq$x,geneNamesByDescCorr))

plot(sortedRsq$x[1:1500],xlab = "Gene or Contig Index (Index(n)=Gene or Contig Name)", ylab = expression(paste("R"^"2")),main="Gene Correlations to 'Indicator Gene' ESR1" )


#this is just exploratory so I'll keep the top 20 R2 and gene or contig name
top20 = data.frame(cbind(sortedRsq$x[1:20],geneNamesByDescCorr[1:20]))
plot(sortedRsq$x[1:20],xlab = "Gene or Contig Index (Index(n)=Gene or Contig Name)", ylab = expression(paste("R"^"2")), main="Gene Correlations to 'Indicator Gene' ESR1")

#compare to different feature selection method
#is colinearity high? if so maybe different method

#_____________________________________________________________________________________________
#set esr1 threshold and compare to M/B diagnosis for comparisons of morphological and genetic activity
#esr1 is known indicator of visceral metastasis
#_____________________________________________________________________________________________
esrLevels = sort.int(esr1SortedGeneData$esr1, partial = NULL, na.last = NA, decreasing = TRUE,
                     method = c("auto", "shell", "quick", "radix"), index.return = TRUE)
plot(esrLevels$x*100,main="ESR1 Expression",xlab="Sample Number",ylab="% Change in Expression Levels")

#what decision boundary or regression best fits best morphological predictor over ESR1 levels

#

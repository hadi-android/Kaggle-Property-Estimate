setwd("D:/Google Drive/Kaggle/Zestimate")
# load the library needed for some basic data processing
library(caret)

#load the labels (property estimates)
tr_lbl = read.csv("train_2016.csv")
#load the training data set
tr_data = read.csv("properties_2016.csv")
summary(tr_data)

# count Nas in each column
NAcount = sapply(tr_data, function(x) sum(is.na(x)))
# compute the fraction of NAs in the column
NAcount_ratio = NAcount/dim(tr_data)[1]
# discard columns (features) with high NA rate (i.e. more than 0.1)
highNA_features = NAcount_ratio > 0.1
highNA_features = subset(highNA_features,highNA_features==T)

names(highNA_features)
tr_data_clean = tr_data
for (i in 1:length(highNA_features)){
  tr_data_clean = tr_data_clean[ , -which(names(tr_data_clean)==names(highNA_features)[i])]
}

#identify and remove features with near zero variance
nearZeroVarFeatures = nearZeroVar(tr_data_clean)
tr_data_clean = tr_data_clean[,-nearZeroVarFeatures]
tr_data_clean = tr_data_clean[complete.cases(tr_data_clean),]

#merge the cleaned data set with the labels
mergedData = merge(tr_data_clean,tr_lbl,by="parcelid")
attach(mergedData)

# load('input_data.RData')
library(xgboost)

# identify properties with multiple time points
dupId = which(duplicated(tr_lbl$parcelid))
unq = tr_lbl[-dupId,]
unq = unq[order(unq$parcelid),]
write.csv(unq, file="training_dup.csv")
N=dim(unq)[1]

#train xgboost
# split training data set into a training and a test set in order to validate the model
inTrain = createDataPartition(mergedData$logerror, p=0.9, list = F)
data_tr = mergedData[inTrain,]
data_te = mergedData[-inTrain,]
model <- xgboost(data = data.matrix(data_tr[,-26]), label = data_tr$logerror, max.depth = 2, eta = 1, nthread = 2, nround = 100)
summary(model)
model

#test xgboost model by prediction on test set
data_te$pred = predict(model,data.matrix(data_te))
predicted = data_te$pred
actual = data_te$logerror
# compute MAE (mean absolute error)
MAE = mean(abs(actual-predicted))
MAE

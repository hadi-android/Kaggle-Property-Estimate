setwd("D:/Google Drive/Kaggle/Zestimate")
# load the library needed for some basic data processing
library(caret)
library(data.table)
# 
#load the labels (property estimates)
tr_lbl = fread("train_2016_v2.csv")
tr_lbl = tr_lbl[order(tr_lbl$parcelid),]
# load the training data set
tr_data = fread("properties_2016.csv")
# summary(tr_data)
samp_sub = fread("sample_submission.csv", header = T)
 
# count Nas in each column
NAcount = sapply(tr_data, function(x) sum(is.na(x)))
# compute the fraction of NAs in the column
NAcount_ratio = NAcount/dim(tr_data)[1]
# discard columns (features) with high NA rate (i.e. more than 0.1)
highNA_features = NAcount_ratio > 0.1
highNA_features = subset(highNA_features,highNA_features==T)
 
names(highNA_features)
tr_data_clean = data.frame(tr_data)
for (i in 1:length(highNA_features)){
  tr_data_clean = tr_data_clean[ , -which(names(tr_data_clean)==names(highNA_features)[i])]
}

# identify and remove features with near zero variance
# nearZeroVarFeatures = nearZeroVar(tr_data_clean)
load('nearZeroVarFeatures.RData')
tr_data_clean = tr_data_clean[,-nearZeroVarFeatures]
# tr_data_clean = tr_data_clean[complete.cases(tr_data_clean),]
# load('input_data_v2.RData')

#get month of transacion
tr_lbl$transactiondate = as.Date(tr_lbl$transactiondate)
tr_lbl$month = format(tr_lbl$transactiondate, "%m")
#
#merge the cleaned data set with the labels
mergedData = merge(tr_data_clean,tr_lbl,by="parcelid")
attach(mergedData)

# free up some memory
# rm(tr_data)
# rm(tr_lbl)
# rm(tr_data_clean)

unique_parcel_merged = data.frame(unique(mergedData$parcelid))
colnames(unique_parcel_merged) = "parcelid"

samp_parcelId = data.frame(samp_sub$ParcelId)
colnames(samp_parcelId) = "parcelid"
common = merge(samp_parcelId, unique_parcel_merged, by="parcelid")

mergedData_unq = mergedData[!duplicated(mergedData$parcelid),]
N1 = dim(unique_parcel_merged)[1]
N1

test_months = c(10,11,12)
N_test_months = length(test_months)
M = length(test_months)
M

# for (i in seq(N_test_months)){
#   tmp = mergedData_unq
#   tmp$month = test_months[i]
#   if(i==1)
#     data_te = tmp
#   else
#     data_te = rbind(data_te, tmp)
# }


# remove irrelevant variables for model prediction
mergedData_parcelid = mergedData$parcelid
mergedData_transdate = mergedData$transactiondate
drops <- c("parcelid", "transactiondate")
mergedData = mergedData[ , !(names(mergedData) %in% drops)]


library(xgboost)

#train xgboost
# split training data set into a training and a test set in order to validate the model
set.seed(1)
inTrain = createDataPartition(mergedData$logerror, p=0.9, list = F)
data_tr_tr = mergedData[inTrain,]
data_tr_te = mergedData[-inTrain,]
model <- xgboost(data = data.matrix(data_tr_tr[,-25]), label = data_tr_tr$logerror, max.depth = 3, eta = 1, nthread = 2, nround = 100)

summary(model)
model

# dtrain <- xgb.DMatrix(data.matrix(data_tr), label = data_tr$logerror)
# cvmodel <- xgb.cv(data = dtrain, nrounds = 100, nthread = 2, nfold = 5, metrics = list("mae"),
# max_depth = 2, eta = 1)
# summary(cvmodel)
# cvmodel
#test xgboost model by prediction on test set
data_tr_te$pred = predict(model,data.matrix(data_tr_te))
predicted = data_tr_te$pred
actual = data_tr_te$logerror
# compute MAE (mean absolute error)
MAE = mean(abs(actual-predicted))
MAE 

#MAE = 0.07537082

# now preidct log error of the full properties data set
# tr_data$month=0
tr_lbl1 = tr_lbl[,-c(2:3)]
data_te = merge(tr_data, tr_lbl1, by="parcelid", all.x = T)
data_te = data_te[!duplicated(data_te$parcelid),]
data_te = merge(samp_parcelId, data_te, by = "parcelid")
# a = unique(data_te$parcelid)
# b=unique(samp_parcelId)

pred_all = predict(model,data.matrix(data_te))
length(pred_all)
dim(tr_data)[1]

a=unique(tr_data$parcelid)

results <- data.table(ParcelId=data_te$parcelid,
'201610'= pred_all,
'201611'= pred_all, 
'201612'= pred_all,
'201710'= pred_all,
'201711' = pred_all,
'201712' = pred_all)

#'201711'= data_te$pred[which(data_te$month==11)],

write.csv(results, file="results.csv", row.names = F)


library(ggplot2)
library(plyr)
library(dplyr)
library(caret)
library(glmnet)
library(moments)

train <- read.csv("C:\\Users\\chinm\\Downloads\\house-prices-advanced-regression-techniques\\train.csv", stringsAsFactors=FALSE)
test <- read.csv("C:\\Users\\chinm\\Downloads\\house-prices-advanced-regression-techniques\\test.csv", stringsAsFactors=FALSE)

head(train)

alldata <- rbind(select(train,MSSubClass:SaleCondition),select(test,MSSubClass:SaleCondition))

train$SalePrice <- log(train$SalePrice + 1)

featurevar <- sapply(names(alldata),function(x){class(alldata[[x]])})
numericvar <-names(featurevar[featurevar != "character"])

skewedvar <- sapply(numericvar,function(x){skewness(alldata[[x]],na.rm=TRUE)})

skewedvar <- skewedvar[skewedvar > 0.7]

for(x in names(skewedvar)) {
  alldata[[x]] <- log(alldata[[x]] + 1)
}

categoricalvar <- names(featurevar[featurevar == "character"])

dummies <- dummyVars(~.,alldata[categoricalvar])
categorical <- predict(dummies,alldata[categoricalvar])
categorical[is.na(categorical)] <- 0  

numeric <- alldata[numericvar]

for (x in numericvar) {
  mean_value <- mean(train[[x]],na.rm = TRUE)
  alldata[[x]][is.na(alldata[[x]])] <- mean_value
}

alldata <- cbind(alldata[numericvar],categorical)

data_train <- alldata[1:nrow(train),]
data_test <- alldata[(nrow(train)+1):nrow(alldata),]
y <- train$SalePrice

cvparam <- trainControl(method="repeatedcv", number=10, repeats=10)

lambdas <- seq(1,0,-0.001)

ridgereg <- train(x=data_train,y=y, method="glmnet", metric="RMSE", trControl=cvparam, tuneGrid=expand.grid(alpha=0, lambda=lambdas))

mean(ridgereg$resample$RMSE)

lassoreg <- train(x=data_train,y=y, method="glmnet", metric="RMSE", trControl=cvparam, tuneGrid=expand.grid(alpha=1,lambda=lambdas))

mean(lassoreg$resample$RMSE)


predicted <- exp(predict(lassoreg,newdata=data_test)) - 1
solution <- data.frame(Id=as.integer(rownames(data_test)),SalePrice=predicted)
write.csv(solution,"finalsubmission2.csv")
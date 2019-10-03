#Homework 8 group work
library(MASS)
library(randomForest)
library(tidyverse)
library(caret)
library(gbm)
library(DataExplorer)
library(recipes)
library(glmnet)
library(rsample)
library(xgboost)
library(mclust)

data_raw = Wholesale

glimpse(data_raw)
plot_missing(data_raw)
plot_density(data_raw)

data_raw = data_raw %>% select(Fresh, everything())

cake = recipe(Fresh~., data=data_raw) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  step_BoxCox(all_numeric(), -all_outcomes()) %>%
  prep(data=data_raw)

cake
data_clean = bake(cake, new_data = data_raw)
modelLookup("svmLinear")
ctrl = trainControl(method="repeatedcv",
                    number=10,
                    repeats=3,
                    classProbs=TRUE,
                    verboseIter=TRUE,
                    savePredictions=TRUE,
                    summaryFunction=defaultSummary)

svm.linear = train(Fresh ~ ., data=data_clean, method="svmLinear",
                   trControl=ctrl, metric="Rsquared",
                   tuneLength=10)
svm.linear

svm.poly = train(Fresh ~ ., data=data_clean, method="svmPoly",
                 trControl=ctrl, metric="RMSE", tunelength=10)
svm.poly

svm.radial = train(Fresh ~ ., data=data_clean, method="svmRadial",
                   trControl=ctrl, metric="Rsquared",
                   tuneLength=10)
svm.radial

glimpse(data_raw)

Fresh = data_raw$Fresh
data_clust = data_raw[,-1]


cluster = Mclust(data_clust,G=1:8)

plot(cluster)
names(cluster)
plot(cluster$classification)

data_clust$CLUST = cluster$classification
data_clust$Fresh = Fresh

glimpse(data_clust)

fresh_C1 = data_clust %>%
  select(Fresh, everything()) %>%
  filter(CLUST==1)

fresh_C2 = data_clust %>%
  select(Fresh, everything()) %>%
  filter(CLUST==2)

fresh_C3 = data_clust %>%
  select(Fresh, everything()) %>%
  filter(CLUST==3)

fresh_C4 = data_clust %>%
  select(Fresh, everything()) %>%
  filter(CLUST==4)

ctrl = trainControl(method="repeatedcv",
                    number=10,
                    repeats=3,
                    classProbs=TRUE,
                    verboseIter=TRUE,
                    savePredictions=TRUE,
                    summaryFunction=defaultSummary)

#CLuster 1
svm.linear = train(Fresh ~ ., data=fresh_C1, method="svmLinear",
                   trControl=ctrl, metric="Rsquared",
                   tuneLength=10)
svm.linear

svm.poly = train(Fresh ~ ., data=fresh_C1, method="svmPoly",
              	trControl=ctrl, metric="Rsquared",
             	tuneLength=10)
svm.poly

svm.radial = train(Fresh ~ ., data=fresh_C1, method="svmRadial",
                   trControl=ctrl, metric="Rsquared",
                   tuneLength=10)
svm.radial

#CLuster 2
svm.linear = train(Fresh ~ ., data=fresh_C2, method="svmLinear",
                   trControl=ctrl, metric="Rsquared",
                   tuneLength=10)
svm.linear

#svm.poly = train(Fresh ~ ., data=fresh_C2, method="svmPoly",
              	#trControl=ctrl, metric="Rsquared",
            #	tuneLength=10)
#svm.poly

svm.radial = train(Fresh ~ ., data=fresh_C2, method="svmRadial",
                   trControl=ctrl, metric="Rsquared",
                   tuneLength=10)
svm.radial

#CLuster 3
svm.linear = train(Fresh ~ ., data=fresh_C3, method="svmLinear",
                   trControl=ctrl, metric="Rsquared",
                   tuneLength=10)
svm.linear

#svm.poly = train(Fresh ~ ., data=fresh_C3, method="svmPoly",
#              	trControl=ctrl, metric="Rsquared",
#             	tuneLength=10)
#svm.poly

svm.radial = train(Fresh ~ ., data=fresh_C3, method="svmRadial",
                   trControl=ctrl, metric="Rsquared",
                   tuneLength=10)
svm.radial

#CLuster 4
svm.linear = train(Fresh ~ ., data=fresh_C4, method="svmLinear",
                   trControl=ctrl, metric="Rsquared",
                   tuneLength=10)
svm.linear

#svm.poly = train(Fresh ~ ., data=fresh_C4, method="svmPoly",
#              	trControl=ctrl, metric="Rsquared",
#             	tuneLength=10)
#svm.poly

svm.radial = train(Fresh ~ ., data=fresh_C4, method="svmRadial",
                   trControl=ctrl, metric="Rsquared",
                   tuneLength=10)
svm.radial

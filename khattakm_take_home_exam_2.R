library(MASS)
library(recipes)
library(rsample)
library(car)
library(DataExplorer)
library(polycor)
library(tidyverse)
library(ROCR)
library(caret)
library(glmnet)
library(randomForest)
library(dplyr)

data_raw <- Telco_Churn
plot_missing(data_raw)
data_raw = na.omit(data_raw)
data_clean <- data_raw %>%
  dplyr::select(Churn, everything()) %>%
  select(-customerID, -PaymentMethod, -PaperlessBilling) 
dim(Telco_Churn)
plot_missing(data_clean)
dim(data_clean)
glimpse(data_clean)

set.seed(1)
train_test_split = initial_split(data_clean, prop=.4)
train_test_split
train_tbl = training(train_test_split)
test_tbl = testing(train_test_split)

#Recipe
cake = recipe(Churn ~ ., data=train_tbl) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  step_BoxCox(all_numeric(), -all_outcomes()) %>%
  prep(data = train_tbl)
cake

# Bake the data
train_clean = bake(cake, new_data=train_tbl)
test_clean = bake(cake, new_data=test_tbl) 
plot_missing(train_clean)

x_train <- model.matrix(Churn ~ ., data_clean)[,-1]
y_train <- as.factor(data_clean$Churn) 

# CV Lasso Model
grid=10^seq(8,-3,length=1000)
ch.lasso <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0.5, lambda = grid)
plot(ch.lasso)

# best lambda
bestlam <- ch.lasso$lambda.min
bestlam
ch.lasso.coef <- predict(ch.lasso, type="coefficients", s = bestlam)[1:27,]
ch.lasso.coef[ch.lasso.coef!=0]
 
# Reduced non-zero coefficicent predictors model here - step 8
Ch.relevant <- data_raw %>%
  select(Churn, tenure, InternetService, OnlineBackup, TechSupport, StreamingMovies, PaperlessBilling, gender, 
         PhoneService, TotalCharges, SeniorCitizen, MultipleLines, OnlineSecurity, DeviceProtection, StreamingTV,
         Contract, Dependents, StreamingTV)
Ch.relevant <- na.omit(Ch.relevant)

set.seed(2)
train_test_split2 <- initial_split(Ch.relevant, prop = 0.4)
train_test_split2

train_tbl_2 <- training(train_test_split2)
test_tbl_2  <- testing(train_test_split2)
glimpse(test_tbl)

aloo <- recipe(Churn ~ ., data = train_tbl_2) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>%
  step_BoxCox(all_numeric(), -all_outcomes()) %>%
  prep(data = train_tbl_2)
aloo

train_clean_2 <- bake(rec_obj, new_data = train_tbl_2)
test_clean_2  <- bake(rec_obj, new_data = test_tbl_2)
glimpse(test_clean_2)

ctrl <- trainControl(method = "repeatedcv", number=10, repeats=3, summaryFunction = twoClassSummary, classProbs = TRUE, savePredictions = "final")
Ch.relevant.log <- train(Churn ~ ., data = train_clean_2, method = "glm", family="binomial", trControl = ctrl, metric = "ROC")
Ch.relevant.log

#Logistic
summary(Ch.relevant.log)
Ch.relevant.logpred = predict(Ch.relevant.log, newdata=test_clean_2)
confusionMatrix(data=Ch.relevant.logpred, test_clean_2$Churn)

#LDA
ctrl <- trainControl(method = "repeatedcv", number=10, repeats=3, 
                     summaryFunction = twoClassSummary, classProbs = TRUE,
                     savePredictions = "final")
Ch.relevant.lda<- train(Churn ~ ., data = train_clean_2,
                      method = "lda", trControl = ctrl, metric = "ROC", preProcess=c('scale', 'center'))
Ch.relevant.lda
summary(Ch.relevant.lda)
Ch.relevant.ldapred = predict(Ch.relevant.lda, newdata=test_clean_2)
confusionMatrix(data=Ch.relevant.ldapred, test_clean_2$Churn)


#QDA #doest work yet

Ch.relevant_2 <- data_raw %>%
  select(Churn, tenure, InternetService, OnlineBackup, TechSupport, StreamingMovies, PaperlessBilling, gender, 
         PhoneService, SeniorCitizen, MultipleLines, OnlineSecurity, DeviceProtection, StreamingTV,
         Dependents)
Ch.relevant_2 <- na.omit(Ch.relevant_2)
#do reciper again
set.seed(3)
train_test_split3 <- initial_split(Ch.relevant_2, prop = 0.4)
train_test_split3

train_tbl_3 <- training(train_test_split3)
train_tbl_3 <- na.omit(train_tbl_3)
test_tbl_3  <- testing(train_test_split3)
test_tbl_3 <- na.omit(test_tbl_3)

potato <- recipe(Churn ~ ., data = train_tbl_3) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>%
  step_BoxCox(all_numeric(), -all_outcomes()) %>%
  prep(data = train_tbl_3)
potato

train_clean_3 <- bake(rec_obj, new_data = train_tbl_3)
test_clean_3  <- bake(rec_obj, new_data = test_tbl_3)
ctrl <- trainControl(method = "repeatedcv", number=10, repeats=3, 
                     summaryFunction = twoClassSummary, classProbs = TRUE,
                     savePredictions = "final")
Ch.relevant.qda<- train(Churn ~ ., data = train_clean_3,
                        method = "qda", trControl = ctrl, metric = "ROC", preProcess=c('scale', 'center'))
Ch.relevant.qda
warnings()
Ch.relevant.ldapred = predict(Ch.relevant.lda, newdata=test_clean_3)
confusionMatrix(data=Ch.relevant.ldapred, test_clean_3$Churn)
hetcor(Ch.relevant_2)


# another try - QDA
qda.fit = train(Churn ~ tenure + SeniorCitizen, data=train_clean_3,
                method="qda", metric="ROC", trControl=ctrl)
print(qda.fit)

qda.pred = predict(qda.fit, newdata=Ch.relevant_2)
confusionMatrix(data=qda.pred, Ch.relevant_2$Churn)

#########
#random forest # step 12
rf.fit = randomForest(Churn ~., data=train_clean_2)
print(rf.fit)
plot(rf.fit)

rf.pred = predict(rf.fit, newdata=test_clean_2)
confusionMatrix(data=rf.pred, test_clean_2$Churn)

#####################################
#Random Forest Standard using caret #Step 13
ctrl <- trainControl(method = "repeatedcv", number=10, repeats=3, 
                     summaryFunction = twoClassSummary, classProbs = TRUE,
                     savePredictions = "final")

set.seed(9) 
Ch.relevant.tree<- train(Churn ~ ., data = train_clean_2,
                         ntree = 80, 
                        method = "rf", trControl = ctrl, metric = "ROC", preProcess=c('scale', 'center'))
Ch.relevant.tree
summary(Ch.relevant.tree)
Ch.relevant.tree.ldapred = predict(Ch.relevant.tree, newdata=test_clean_2)
confusionMatrix(data=Ch.relevant.tree.ldapred, test_clean_2$Churn)


#####################################
#cross validated gradient boosted using caret
#step 14
ctrl <- trainControl(method = "repeatedcv", number=10, repeats=3, 
                     summaryFunction = twoClassSummary, classProbs = TRUE,
                     savePredictions = "final")

Ch.relevant.gbm <- train(Churn ~ . , method = "gbm", data = train_clean_2, 
                   trControl = ctrl, metric = "ROC", 
                   verbose = FALSE, tuneLength = 5)
Ch.relevant.gbm

Ch.relevant.gbm.pred <- predict(Ch.relevant.gbm, newdata=test_clean_2)
confusionMatrix(data=Ch.relevant.gbm.pred, test_clean_2$Churn)

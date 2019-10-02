#take home exam
#khattakm
library(caret)
library(e1071)
library(kernlab)
library(recipes)
library(xgboost)
library(randomForest)
library(tidyverse)
library(DataExplorer)
library(polycor)
library(rsample)
library(caretEnsemble)
library(glmnet)
modelLookup("svmLinear")
modelLookup("svmPoly")
modelLookup("svmRadial")

IncomeData <- Income_Pred
Income_data <- IncomeData[1:5000, ]
glimpse(Income_data)
Income_data[Income_data== " ?"] = NA
Income_data[Income_data== "?"] = NA
levels(Income_data$Employer)[levels(Income_data$Employer)==" ?"] = NA
levels(Income_data$Employer)
levels(Income_data$JobRole)[levels(Income_data$JobRole)==" ?"] = NA
Income_data$JobRole
levels(Income_data$JobRole)
glimpse(Income_data)

Income_data$Income <- ifelse(Income_data$Income ==" <=50K", "Bad", "Good")
Income_data$Income <- as.factor(Income_data$Income)
Income_datum <- Income_data %>%
  dplyr::select(Income, everything()) %>%
  dplyr::select(-Country, -FamilyRole) #due to high correlation with Sex

IncomeData <- na.omit(Income_datum)
plot_density(IncomeData)
glimpse(IncomeData)
plot_missing(IncomeData)
summary(IncomeData)
hetcor(IncomeData)

gulab <- recipe(Income~., data=IncomeData) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  #step_bagimpute(all_nominal(), -all_outcomes()) %>%
  step_YeoJohnson(all_numeric(), -all_outcomes()) %>%
  step_nzv(all_predictors())%>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>%
  prep(data = IncomeData)
data_clean <- bake(gulab, new_data = IncomeData)

glimpse(data_clean)
plot_missing(data_clean)

set.seed(4557)
train_x <-  model.matrix(Income ~ . -1, data = data_clean)
train_y <-  data_clean$Income
grid = 10^seq(10,-2,by=-.1)
cv.lasso <-  cv.glmnet(train_x, train_y, family="binomial", alpha=1, lambda=grid)
plot(cv.lasso)
best_lambda = cv.lasso$lambda.min
coef(cv.lasso)

relevant_clean = data_clean %>%
  dplyr::select(Income, Age, EducationYears, WeeklyHours, Employer_X.Self.emp.inc, 
                Employer_X.Self.emp.not.inc, MaritalStatus_X.Married.civ.spouse, 
                JobRole_X.Exec.managerial, JobRole_X.Farming.fishing,
                JobRole_X.Other.service, JobRole_X.Prof.specialty)
glimpse(relevant_clean)

control <- trainControl( method= "cv", 
                         number = 10,
                         verboseIter = TRUE,
                         classProbs = TRUE, savePredictions = TRUE)
algorithmList = c('lda', 'gbm', 'knn', 'svmRadial')

set.seed(85)
models <- caretList(Income~., data=relevant_clean, trControl = control, methodList = algorithmList)
results <- resamples(models)
summary(results)
dotplot(results)

#correlation between results
modelCor(results)
splom(results)

#stack using glm
set.seed(12)
stack.glm <- caretStack(models, method = "glm", metric = "Accuracy", trControl = control)
print(stack.glm)

#stack using rf
set.seed(13)
stack.rf <- caretStack(models, method = "rf", metric = "Accuracy", trControl = control)
print(stack.rf)

#stack using xgb
set.seed(14)
stack.xgboost <- caretStack(models, method = "xgbTree", metric = "Accuracy", trControl = control)
print(stack.xgboost)


#for step 6
algorithmList2 = c('rf', 'gbm', 'ada', 'svmRadial')
set.seed(84)
models2 <- caretList(Income~., data=relevant_clean, trControl = control, methodList = algorithmList2)
results2 <- resamples(models2)
summary(results2)
dotplot(results2)

#correlation between results
modelCor(results2)
splom(results2)

#stack using glm
set.seed(17)
stack.glm2 <- caretStack(models2, method = "glm", metric = "Accuracy", trControl = control)
print(stack.glm2)

#stack using rf
set.seed(19)
stack.rf2 <- caretStack(models2, method = "rf", metric = "Accuracy", trControl = control)
print(stack.rf2)

#stack using xgb
set.seed(18)
stack.xgboost2 <- caretStack(models2, method = "xgbTree", metric = "Accuracy", trControl = control)
print(stack.xgboost2)


#for step 7
#MODEL 1
algorithmList3 = c('rf', 'svmRadial')
set.seed(100)
models3 <- caretList(Income~., data=relevant_clean, trControl = control, methodList = algorithmList2)
results3 <- resamples(models3)
summary(results3)
dotplot(results3)
modelCor(results3)
splom(results3)
set.seed(101)
stack.glm3 <- caretStack(models3, method = "glm", metric = "Accuracy", trControl = control)
print(stack.glm3)


#MODEL2
algorithmList4 = c('glm', 'C5.0')
set.seed(1000)
models4 <- caretList(Income~., data=relevant_clean, trControl = control, methodList = algorithmList2)
results4 <- resamples(models4)
summary(results4)
dotplot(results4)
modelCor(results4)
splom(results4)
set.seed(1010)
stack.glm4 <- caretStack(models4, method = "rf", metric = "Accuracy", trControl = control)
print(stack.glm4)

#MODEL3
algorithmList5 = c('glm', 'ada')
set.seed(10000)
models5 <- caretList(Income~., data=relevant_clean, trControl = control, methodList = algorithmList2)
results5 <- resamples(models5)
summary(results5)
dotplot(results5)
modelCor(results5)
splom(results5)
set.seed(10100)
stack.glm5 <- caretStack(models5, method = "C5.0", metric = "Accuracy", trControl = control)
print(stack.glm5)

#MODEL4
algorithmList6 = c('rf', 'C5.0')
set.seed(100000)
models6 <- caretList(Income~., data=relevant_clean, trControl = control, methodList = algorithmList2)
results6 <- resamples(models6)
summary(results6)
dotplot(results6)
modelCor(results6)
splom(results6)
set.seed(101000)
stack.glm6 <- caretStack(models6, method = "glm", metric = "Accuracy", trControl = control)
print(stack.glm6)
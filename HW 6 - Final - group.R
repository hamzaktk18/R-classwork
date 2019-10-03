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

data_raw = HR_Churn
glimpse(data_raw)

plot_missing(data_raw)
plot_density(data_raw)

set.seed(993)
train_test_split = initial_split(data_raw, prop=.70)
train_test_split

train_tbl = training(train_test_split)
test_tbl = testing(train_test_split)

cake = recipe(Gone ~., data=train_tbl) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot=TRUE) %>%
  step_BoxCox(all_predictors(), -all_outcomes()) %>%
  prep(data=train_tbl)

cake

train_clean = bake(cake,new_data=train_tbl)
test_clean = bake(cake,new_data=test_tbl)

train_clean$Gone = as.factor(train_clean$Gone)
test_clean$Gone = as.factor(test_clean$Gone)
glimpse(train_clean)

set.seed(3432)
train_x <-  model.matrix(Gone ~ . -1, data = train_clean)
train_y <-  train_clean$Gone
grid = 10^seq(10,-2,by=-.1)
cv.lasso <-  cv.glmnet(train_x, train_y, family="binomial", alpha=1, lambda=grid)
plot(cv.lasso)

best_lambda = cv.lasso$lambda.min
coef(cv.lasso)

train_clean2 = train_clean %>% select(-PerformanceRating,-RelationshipSatisfaction,-YearsAtCompany,-YearsSinceLastPromotion)
glimpse(train_clean2)
test_clean2 = test_clean %>% select(-PerformanceRating,-RelationshipSatisfaction,-YearsAtCompany,-YearsSinceLastPromotion)

set.seed(3854)
train_x <-  model.matrix(Gone ~ . -1, data = train_clean2)
train_y <-  train_clean2$Gone
grid = 10^seq(10,-2,by=-.1)
cv.lasso2 <-  cv.glmnet(train_x, train_y, family="binomial", alpha=1, lambda=grid)
plot(cv.lasso2)
###########################
lasso.prob =  predict(cv.lasso2, s = best_lambda, newx = data.matrix(train_x), type="response")
lasso.pred = ifelse(lasso.prob>0.5, "Yes", "No")
lasso.prob
lasso.pred

ConfusionMatrix(table(lasso.pred, test_y))
mean(lasso.pred==test_clean2$Gone)
###################

best_lambda = cv.lasso2$lambda.min
coef(cv.lasso2)
best_lambda


test_x <-  model.matrix(Gone ~ . -1, data = test_clean2)
test_y <-  test_clean2$Gone


lasso.pred = predict(cv.lasso2, newx = data.matrix(train_x), type="response")

lasso.pred
                     
control = trainControl(method="repeatedcv", number=10, repeats=3, summaryFunction=twoClassSummary,classProbs=TRUE, savePredictions="final")

lda.fit = train(Gone ~., data=train_clean2, method="lda", metric="ROC", trControl=control)
lda.fit

lda.pred = predict(lda.fit, test_clean2)
confusionMatrix(lda.pred, test_clean2$Gone)




#grid = expand.grid(alpha=1, lambda=10^seq(10,-2,length=1000))
#control = trainControl(method="repeatedcv", number=10, repeats=3)
#cv.lasso = train(Gone ~., data=train_clean,
#method="glmnet",
#trControl=control,
#tuneGrid=grid)

#cv.lasso$lambda.min
#coef(cv.lasso, s=0.1)

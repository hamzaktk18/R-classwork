library(MASS)
library(recipes)
library(rsample)
library(car)
library(DataExplorer)
library(polycor)
library(tidyverse)
library(ROCR)
library(caret)

data("Boston")
data_raw = Boston
glimpse(data_raw)

data_raw = data_raw %>% select(medv,everything())

plot_density(data_raw)
hetcor(data_raw)

pancakes = recipe(medv ~ ., data=data_raw) %>%
  step_BoxCox(all_predictors(), -all_outcomes()) %>%
  prep(data=data_raw)

pancakes

data_clean = bake(pancakes, new_data=data_raw)

#First linear model
set.seed(20000)
train_test_split = initial_split(data_clean, prop=0.70)
train_test_split

train_clean = training(train_test_split)
dim(train_clean)
test_clean = testing(train_test_split)
dim(test_clean)

lm.fit = lm(medv ~ ., data=train_clean)
summary(lm.fit)

#Second linear model
set.seed(4534)
train_test_split = initial_split(data_clean, prop=0.70)
train_test_split

train_clean2 = training(train_test_split)
test_clean2 = testing(train_test_split)

lm.fit2 = lm(medv ~ ., data=train_clean2)
summary(lm.fit2)

#Third linear model
set.seed(76574)
train_test_split = initial_split(data_clean, prop=0.70)
train_test_split

train_clean3 = training(train_test_split)
test_clean3 = testing(train_test_split)

lm.fit3 = lm(medv ~ ., data=train_clean3)
summary(lm.fit3)

#Fourth linear model
set.seed(9375)
train_test_split = initial_split(data_clean, prop=0.70)
train_test_split

train_clean4 = training(train_test_split)
test_clean4 = testing(train_test_split)

lm.fit4 = lm(medv ~ ., data=train_clean4)
summary(lm.fit4)

#Fifth linear model
set.seed(1)
train_test_split = initial_split(data_clean, prop=0.70)
train_test_split

train_clean5 = training(train_test_split)
test_clean5 = testing(train_test_split)

lm.fit5 = lm(medv ~ ., data=train_clean5)
summary(lm.fit5)

#Sixth linear model
set.seed(986754)
train_test_split = initial_split(data_clean, prop=0.70)
train_test_split

train_clean6 = training(train_test_split)
test_clean6 = testing(train_test_split)

lm.fit6 = lm(medv ~ ., data=train_clean6)
summary(lm.fit6)

#cross validated linear model
set.seed(2500)
train_test_split = initial_split(data_clean, prop=0.70)
train_test_split

B_train_clean = training(train_test_split)
B_test_clean = testing(train_test_split)

# Build the linear regression model with embedded cross validation

ctrl<-trainControl(method = "repeatedcv", number = 10, repeats = 3, summaryFunction = defaultSummary)
BTown.cv.fit <- train(medv ~ ., data = B_train_clean , method = "lm", trControl = ctrl, metric= "Rsquared")

summary(BTown.cv.fit)


predCV = predict(BTown.cv.fit, B_test_clean)
modelvalues <- data.frame(obs = B_test_clean$medv, pred=predCV)
defaultSummary(modelvalues)
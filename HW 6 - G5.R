library(MASS)
library(mlbench)
library(caret)
library(DataExplorer)
library(tidyverse)
library(polycor)
library(car)
library(rsample)
library(recipes)
library(ROCR)
library(broom)
library(class)
library(glmnet)

data_raw = HR_Churn
dim(data_raw)
glimpse(data_raw)

plot_missing(data_raw)
hetcor(data_raw)



#Split
set.seed(6875)
train_test_split = initial_split(data_raw, prop=0.8)
train_test_split

train_clean = training(train_test_split)
test_clean = testing(train_test_split)

#recipe
rec_obj = recipe(Gone ~ ., data=data_raw) %>%
  step_BoxCox(all_numeric(), -all_outcomes()) %>% #to normalize the thingy
  prep(data=data_raw)

rec_obj

data_clean = bake(pancakes, new_data=data_raw)

set.seed(1)
lasso = glmnet(x[data_clean,], y[data_clean], alpha=1, lambda =grid)
plot(lasso)


x <- model.matrix(Gone~., data_raw)[,-1]
y <- data_raw$Gone
# = data_raw %>%
#select(Gone) %>%
#unlist() %>%
# as.numeric()

library(mlbench)
library(tidyverse)
library(ISLR)
library(dplyr)
library(DataExplorer)
library(polycor)
library(recipes)
library(rsample)
library(car)
library(caret)
library(ROCR)
library(MASS)
library(broom)
library(class)

HW4Data <- BC
glimpse(HW4Data)

plot_missing(HW4Data)

HW4Data <- HW4Data %>%      
  select(Class, everything())  

hetcor(HW4Data) #for correlation
#Computes a heterogenous correlation matrix, consisting of Pearson product-moment correlations between numeric variables, polyserial correlations between numeric and ordinal variables, and polychoric correlations between ordinal variables.
glimpse(HW4Data)


#splitting the datasets randomly
set.seed(2019)
train_test_split <- initial_split(HW4Data, prop = 0.5)
train_test_split

train_tbl <- training(train_test_split)
test_tbl <- testing(train_test_split)
train <- training(train_test_split)
test  <- testing(train_test_split)

dim(train)
dim(test)


#Retrieve train and test sets
rec_obj <- recipe(Class ~ ., data = train_tbl) %>%
  step_bagimpute(all_predictors(), -all_outcomes()) %>%
  #step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  prep(data = train_tbl)

#Print the recipe object
rec_obj

train_clean <- bake(rec_obj, new_data=train_tbl)
test_clean <- bake(rec_obj, new_data=test_tbl)

glimpse(train_clean)
plot_missing(train_clean)

#Build a logistic model
log.fit <- glm(Class ~ ., data=train_clean, family = binomial) # binomial is binary classification
summary(log.fit)


#Build a Second logistic model
log.fit <- glm(Class ~ Cl.thickness + Cell.shape + Marg.adhesion + Bl.cromatin + Mitoses, data=train_clean, family = binomial) # binomial is binary classification
summary(log.fit)

log.prob = predict(log.fit, newdata=test_clean, type='response')
log.pred = ifelse(log.prob > 0.5, "Malignant", "benign")


table(log.pred, test_clean$Class)
mean(log.pred==test_clean$Class)

plot(log.fit, which=4, id.n=4)

train_clean <- train_clean[-2, ]
train_clean <- train_clean[-18, ]
train_clean <- train_clean[-114, ]
train_clean <- train_clean[-173, ]

durbinWatsonTest(log.fit)
anova(log.fit, test="Chisq") #helps generate a kia-squared or something


#For ROC and AUC
i <- predict(log.fit, newdata=test_clean, type="response") #Not even that important
ir <- prediction(i, test_clean$Class)
irf <- performance(ir, measure = "tnr", x.measure = "fnr")
plot(irf)

aur <- performance(ir, measure = "auc")
auc <- aur@y.values[[1]]
auc


#LDA Model
lda.fit = lda(Class ~ Cl.thickness + Cell.shape + Marg.adhesion + Bl.cromatin + Mitoses, data=train_clean)
summary(lda.fit)
names(lda.fit)

plot(lda.fit)

lda.pred = predict(lda.fit, test_clean)
names(lda.pred)

lda.class = lda.pred$class

table(lda.class,test_clean$Class)
mean(lda.class==test_clean$Class)

#KNN Model


#nn3 <- knn(train_clean, test_clean, cl, k=3)
#table(test_clean[,'Class'],nn3)





#if we had QDA just in case
qda.fit = qda(Class ~ Cl.thickness + Cell.shape + Marg.adhesion + Bl.cromatin + Mitoses, data=train_clean)
summary(qda.fit)
names(qda.fit)

plot(qda.fit)

qda.pred = predict(qda.fit, test_clean)
names(qda.pred)

qda.class = qda.pred$class

table(qda.class,test_clean$Class)
mean(qda.class==test_clean$Class)



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

data("BreastCancer")
data_raw = BreastCancer

view(data_raw)

dim(data_raw)
glimpse(data_raw)

data_raw = data_raw %>% select(-Id)

# convert to numeric##################

for(i in 1:9) {
  data_raw[, i] <- as.numeric(as.character(data_raw[, i]))
}

# Change Y values to 1's and 0's#########################

data_raw$Class <- ifelse(data_raw$Class == "malignant", 1, 0)
data_raw$Class <- factor(data_raw$Class, levels = c(0, 1))


#class(data_raw[[5]])

data_raw = data_raw %>%
  select(Class,everything())
#select(-Id)

plot_missing(data_raw)
hetcor(data_raw)

#data_new = log(data_raw)

set.seed(2019)
train_test_split = initial_split(data_raw, prop=0.5)
train_test_split

train_tbl = training(train_test_split)
test_tbl = testing(train_test_split)

rec_obj = recipe(Class ~ ., data=train_tbl) %>%
  step_bagimpute(all_predictors(), -all_outcomes()) %>%
  prep(data=train_tbl)

rec_obj

train_clean = bake(rec_obj, new_data=train_tbl)
test_clean = bake(rec_obj, new_data=test_tbl)

plot_missing(train_clean)
glimpse(train_clean)

log.fit1 = glm(Class ~ ., data=train_clean, family=binomial, maxit=100)
summary(log.fit1)

vif(log.fit1)
anova(log.fit1, test="Chisq")

log.prob1 = predict(log.fit1, newdata=test_clean, type="response")
log.pred1 = ifelse(log.prob1>0.50,"1","0")

table(log.pred1,test_clean$Class)
mean(log.pred1==test_clean$Class)

log.fit2 = glm(Class ~ Cl.thickness + Cell.shape + Marg.adhesion + Bl.cromatin, data=train_clean, family=binomial, maxit=100)
summary(log.fit2)

log.prob2 = predict(log.fit2, newdata=test_clean, type="response")
log.pred2 = ifelse(log.prob2>0.50,"1","0")

table(log.pred2,test_clean$Class)
mean(log.pred2==test_clean$Class)

plot(log.fit2, which=4, id.n=4)

#######################################################################

train_clean2 = train_clean[-2,]
train_clean2 = train_clean[-19,]
train_clean2 = train_clean[-158,]
train_clean2 = train_clean[-226,]

test_clean2 = test_clean[-2,]
test_clean2 = test_clean[-19,]
test_clean2 = test_clean[-158,]
test_clean2 = test_clean[-226,]

log.fit3 = glm(Class ~ Cl.thickness + Cell.shape + Marg.adhesion + Bl.cromatin, data=train_clean2, family=binomial, maxit=100)
summary(log.fit3)

plot(log.fit3, which=4, id.n=4)

log.prob3 = predict(log.fit3, newdata=test_clean2, type="response")
log.pred3 = ifelse(log.prob3>0.50,"1","0")

table(log.pred3,test_clean2$Class)
mean(log.pred3==test_clean2$Class)

pr = prediction(log.prob3, test_clean2$Class)
prf = performance(pr,measure="tnr",x.measure="fnr")
plot(prf)

auc = performance(pr, measure="auc")
auc = auc@y.values[[1]]
auc

########################################################################
lda.fit = lda(Class ~  Cl.thickness + Cell.shape + Marg.adhesion + Bl.cromatin, data=train_clean2)
summary(lda.fit)

lda.pred = predict(lda.fit, test_clean)
names(lda.pred)

lda.class = lda.pred$class

table(lda.class,test_clean$Class)
mean(lda.class==test_clean$Class)

##########################################################################
train_y = train_clean$Class
glimpse(train_y)

test_y = test_clean$Class

train_clean3 = train_clean[,-1]
test_clean3 = test_clean[-1]

glimpse(train_clean3)

set.seed(1)
knn.pred = knn(as.matrix(train_clean3),as.matrix(test_clean3),as.vector(train_y), k=50)
table(knn.pred,test_y)
mean(knn.pred==test_y)


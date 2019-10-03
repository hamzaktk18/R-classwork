library(ISLR)
library(mlbench)
library(caret)
library(MASS)
library(DataExplorer)
library(tidyverse)
library(polycor)
library(car)
library(recipes)
library(rsample)
library(ROCR)
library(broom)

# load the data into objects
Freedom <- Human_Freedom

#check the dimension and structure of data
glimpse(Freedom)

# move variable to the first column
FreedomHuman <- Freedom %>%
  select(hf_score, 
         pf_ss_disappearances_disap, 
         pf_ss_disappearances_violent,
         pf_ss_disappearances_organized,  
         pf_ss_disappearances_fatalities, 
         pf_score, 
         ef_legal_courts,
         ef_legal_protection,
         ef_legal_military,
         ef_legal_integrity,
         ef_legal_enforcement,
         ef_legal_restrictions,
         ef_legal_police,
         ef_legal_crime,
         ef_legal_gender)

glimpse(FreedomHuman)
#check for missing values by row and column in the data
plot_missing(FreedomHuman)

set.seed(2000)
train_test_split <- initial_split(FreedomHuman, prop = .5)

Htrain <- training(train_test_split)
Htest  <- testing(train_test_split)
dim(Htest)
dim(Htrain)

rec_object <- recipe(hf_score~ ., data = Htrain) %>%
  step_bagimpute(all_predictors(), -all_outcomes()) %>%
  step_BoxCox(all_numeric()) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  prep(data= Htrain)

rec_object
#rlang::last_error()

train_clean <- bake(rec_object, new_data=Htrain)
test_clean <- bake(rec_object, new_data=Htest)

glimpse(train_clean)

#Building a linear  regression model
lm.fit<-lm(hf_score ~., data=train_clean)
summary(lm.fit)

NewFreedomHuman <- train_clean %>%
  select(hf_score, 
         pf_ss_disappearances_organized, 
         pf_score, 
         ef_legal_protection, 
         ef_legal_military, 
         ef_legal_integrity, 
         ef_legal_enforcement, 
         ef_legal_restrictions)

glimpse(NewFreedomHuman)

NewFreedomHuman= as.data.frame(NewFreedomHuman) # Error; argument must be a data frame.
hetcor(NewFreedomHuman)
plot_scatterplot(NewFreedomHuman, by="hf_score")

lm.fit2= lm(hf_score ~ pf_ss_disappearances_organized + pf_score + ef_legal_protection + ef_legal_military + ef_legal_integrity + ef_legal_enforcement+ ef_legal_restrictions, data = Htrain)
summary(lm.fit2)

#Non-Linearity of the Data
crPlots(lm.fit2)

#Heteroscedasticity
ncvTest(lm.fit2)
spreadLevelPlot(lm.fit2)

# Normality of Residuals
qqPlot(lm.fit2, main="QQ Plot")

# normal distribution of predictors test
shapiro.test(Human_Freedom$pf_ss_disappearances_organized)
shapiro.test(Human_Freedom$pf_score)
shapiro.test(Human_Freedom$ef_legal_protection)
shapiro.test(Human_Freedom$ef_legal_military)
shapiro.test(Human_Freedom$ef_legal_integrity)
shapiro.test(Human_Freedom$ef_legal_enforcement)
shapiro.test(Human_Freedom$ef_legal_restrictions)

# Autocorrelation of Error Terms
durbinWatsonTest(lm.fit2)

#for mulitcollinearity
vif(lm.fit2)

#Outliers
outlierTest(lm.fit2)

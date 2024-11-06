library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(embed)
library(naivebayes)
library(discrim)

#Read Data

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/ggg_kaggle")

train <- vroom("train.csv")
train2 <- vroom("trainWithMissingValues.csv")
test <- vroom("test.csv")


my_recipe <- recipe(type~., data = train2) %>% 
  step_impute_bag(hair_length, impute_with = imp_vars(has_soul, color), trees = 250) %>% 
  step_impute_bag(rotting_flesh, impute_with = imp_vars(has_soul, color, hair_length), trees = 250) %>% 
  step_impute_bag(bone_length, impute_with = imp_vars(has_soul, color, hair_length, rotting_flesh), trees = 250)

prepped <- prep(my_recipe)
baked <- bake(prepped, new_data = NULL)

rmse_vec(train[is.na(train2)],baked[is.na(train2)])

  

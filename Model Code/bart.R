library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(embed)
library(naivebayes)
library(discrim)
library(bonsai)
library(dbarts)
library(lightgbm)

#Read Data

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/ggg_kaggle")

train <- vroom("train.csv")
test <- vroom("test.csv")

my_recipe <- recipe(type~., data = train) %>% 
  step_mutate_at(color, fn = factor) 


#Naive Bayes Model

bart_model <- parsnip::bart(trees=tune()) %>% 
  set_mode("classification") %>% 
  set_engine("dbarts")

bart_workflow <- workflow() %>% 
  add_model(bart_model) %>% 
  add_recipe(my_recipe)

tuning_grid <- grid_regular(trees(), levels = 20)

folds <- vfold_cv(train, v = 10, repeats=1)

cv_results <- bart_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid, 
            metrics = metric_set(accuracy))

best_tune <- cv_results %>% select_best(metric='accuracy')


final_workflow <- bart_workflow %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train)

bart_preds <- predict(final_workflow, 
                         new_data = test,
                         type = 'class')

bart_submission <- bart_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_class) %>% 
  rename(type = .pred_class) 


vroom_write(x=bart_submission, file="./Submissions/BartPreds1.csv", delim=",")

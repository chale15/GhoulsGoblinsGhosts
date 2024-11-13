library(kernlab)
library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(embed)
library(naivebayes)
library(discrim)

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/ggg_kaggle")

train <- vroom("train.csv")
test <- vroom("test.csv")

my_recipe <- recipe(type~., data = train) %>% 
  step_mutate_at(color, fn = factor) 

##SVM
svm_radial <- svm_rbf(rbf_sigma=tune(),
                      cost=tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kernlab")

radial_workflow <- workflow() %>% 
  add_model(svm_radial) %>% 
  add_recipe(my_recipe)

tuning_grid_radial <- grid_regular(rbf_sigma(), cost(), levels = 10)

radial_models <- radial_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid_radial, 
            metrics = metric_set(roc_auc, accuracy))

best_tune_radial <- radial_models %>% select_best(metric='accuracy')

final_workflow_radial <- radial_workflow %>% 
  finalize_workflow(best_tune_radial) %>% 
  fit(data = train)

radial_svm_preds <- predict(final_workflow_radial, 
                            new_data = test,
                            type = 'class')

svm_submission <- radial_svm_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_class) %>% 
  rename(type = .pred_class) 


vroom_write(x=svm_submission, file="./Submissions/RadialSVMPreds6.csv", delim=",")


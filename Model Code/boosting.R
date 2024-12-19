library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(embed)
library(naivebayes)
library(discrim)
library(bonsai)
library(lightgbm)

#Read Data

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/ggg_kaggle")

train <- vroom("train.csv")
test <- vroom("test.csv")

my_recipe <- recipe(type~., data = train) %>% 
  step_mutate_at(color, fn = factor) 


#Naive Bayes Model

boosted_model <- boost_tree(tree_depth=tune(),
                        trees=tune(),
                        learn_rate = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("lightgbm")

boosted_workflow <- workflow() %>% 
  add_model(boosted_model) %>% 
  add_recipe(my_recipe)

tuning_grid <- grid_regular(tree_depth(), trees(), learn_rate(), levels = 20)

folds <- vfold_cv(train, v = 10, repeats=1)

cv_results <- boosted_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid, 
            metrics = metric_set(accuracy))

best_tune <- cv_results %>% select_best(metric='accuracy')


final_workflow <- boosted_workflow %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train)

boosted_preds <- predict(final_workflow, 
                    new_data = test,
                    type = 'class')

boosted_submission <- boosted_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_class) %>% 
  rename(type = .pred_class) 


vroom_write(x=boosted_submission, file="./Submissions/BoostedPreds2.csv", delim=",")

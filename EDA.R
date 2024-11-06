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


#my_recipe <- recipe(type~., data = train2) %>% 
#  step_impute_bag(hair_length, impute_with = imp_vars(has_soul, color), trees = 250) %>% 
#  step_impute_bag(rotting_flesh, impute_with = imp_vars(has_soul, color, hair_length), trees = 250) %>% 
#  step_impute_bag(bone_length, impute_with = imp_vars(has_soul, color, hair_length, rotting_flesh), trees = 250)

#prepped <- prep(my_recipe)
#baked <- bake(prepped, new_data = NULL)

#rmse_vec(train[is.na(train2)],baked[is.na(train2)])

my_recipe <- recipe(type~., data = train) %>% 
  step_mutate_at(color, fn = factor)

#Naive Bayes Model

nb_model <- naive_Bayes(Laplace=tune(),
                        smoothness=tune()) %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")

nb_workflow <- workflow() %>% 
  add_model(nb_model) %>% 
  add_recipe(my_recipe)

tuning_grid <- grid_regular(Laplace(), smoothness(), levels = 20)

folds <- vfold_cv(train, v = 10, repeats=1)

cv_results <- nb_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid, 
            metrics = metric_set(accuracy))

best_tune <- cv_results %>% select_best(metric='accuracy')

final_workflow <- nb_workflow %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train)

nb_preds <- predict(final_workflow, 
                    new_data = test,
                    type = 'class')

nb_submission <- nb_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_class) %>% 
  rename(type = .pred_class) 

vroom_write(x=nb_submission, file="./Submissions/NBPreds13.csv", delim=",")
  

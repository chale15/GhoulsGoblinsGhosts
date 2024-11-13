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
library(stacks)
library(rpart)
library(ranger)
library(kernlab)
library(nnet)

#Read Data

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/ggg_kaggle")

train <- vroom("train.csv")
test <- vroom("test.csv")

my_recipe <- recipe(type~., data = train) %>% 
  step_mutate_at(color, fn = factor) 

untunedModel <- control_stack_grid() 
tunedModel <- control_stack_resamples()


## Boosted Model

boosted_model <- boost_tree(tree_depth=tune(),
                            trees=tune(),
                            learn_rate = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("lightgbm")

boosted_workflow <- workflow() %>% 
  add_model(boosted_model) %>% 
  add_recipe(my_recipe)

boosted_tuning_grid <- grid_regular(tree_depth(), trees(), learn_rate(), levels = 20)

folds <- vfold_cv(train, v = 10, repeats=1)

boosted_models <- boosted_workflow %>% 
  tune_grid(resamples = folds,
            grid = boosted_tuning_grid, 
            metrics = metric_set(accuracy, roc_auc),
            control = untunedModel)


## Naive Bayes Model

nb_model <- naive_Bayes(Laplace=tune(),
                        smoothness=tune()) %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")

nb_workflow <- workflow() %>% 
  add_model(nb_model) %>% 
  add_recipe(my_recipe)

nb_tuning_grid <- grid_regular(Laplace(), smoothness(), levels = 20)

folds <- vfold_cv(train, v = 10, repeats=1)

nb_models <- nb_workflow %>% 
  tune_grid(resamples = folds,
            grid = nb_tuning_grid, 
            metrics = metric_set(accuracy, roc_auc),
            control = untunedModel)

## Regression Tree Model

rt_model<- decision_tree(tree_depth=tune(),
                         cost_complexity = tune(),
                         min_n = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

rt_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(rt_model)

rt_tuning_grid <- grid_regular(tree_depth(), cost_complexity(), min_n(), levels = 5)

rt_models <- rt_workflow %>% 
  tune_grid(resamples=folds,
            grid=rt_tuning_grid,
            metrics=metric_set(accuracy, roc_auc),
            control = untunedModel)

rt_best_tune <- rt_models %>% select_best(metric="accuracy")

rt_final_workflow <- rt_workflow %>% 
  finalize_workflow(rt_best_tune) %>% 
  fit(data = train)

rt_preds <- predict(rt_final_workflow, 
                    new_data = test,
                    type = 'class')

rt_submission <- rt_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_class) %>% 
  rename(type = .pred_class) 


vroom_write(x=rt_submission, file="./Submissions/RTPreds1.csv", delim=",")

#Random Forest
rf_model <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

rf_workflow <- workflow() %>% 
  add_recipe(my_recipe_multi) %>% 
  add_model(rf_model)

rf_tuning_grid <- grid_regular(mtry(range=c(1, 11)), min_n(), levels = c(11, 10))

rf_models <- rf_workflow %>%
  tune_grid(resamples=folds,
            grid=rf_tuning_grid,
            metrics=metric_set(roc_auc, accuracy),
            control = untunedModel)

rf_best_tune <- rf_models %>% select_best(metric="accuracy")

rf_final_workflow <- rf_workflow %>% 
  finalize_workflow(rf_best_tune) %>% 
  fit(data = train)

rf_preds <- predict(rf_final_workflow, 
                    new_data = test,
                    type = 'class')

rf_submission <- rf_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_class) %>% 
  rename(type = .pred_class) 


vroom_write(x=rf_submission, file="./Submissions/RFPreds1.csv", delim=",")

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
            metrics = metric_set(roc_auc, accuracy),
            control = untunedModel)

## Multinomial Model

my_recipe_multi <- recipe(type~., data = train) %>% 
  step_mutate_at(color, fn = factor) %>% 
  step_range(all_numeric_predictors(), min=0, max=1)

multi_model <- multinom_reg(penalty=tune()) %>% 
  set_mode("classification") %>% 
  set_engine("nnet")

multi_wf <- workflow() %>% 
  add_model(multi_model) %>% 
  add_recipe(my_recipe_multi)

tuning_grid_multi <- grid_regular(penalty(), levels = 100)
folds <- vfold_cv(train, v = 10, repeats=1)

multi_models <- multi_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid_multi, 
            metrics = metric_set(roc_auc, accuracy),
            control = untunedModel)


## Stack Models

my_stack <- stacks() %>% 
  add_candidates(nb_models) %>% 
  add_candidates(multi_models)
#  add_candidates(radial_models) %>% 
#  add_candidates(boosted_models)
#  add_candidates(rt_models) %>% 
#  add_candidates(rf_models)

my_stack <- my_stack %>% 
  add_candidates(boosted_models)

stack_model <- my_stack %>% 
  blend_predictions() %>% 
  fit_members()

final_preds <- stack_model %>% predict(new_data=test,
                                       type='class')

#Kaggle Submission
stacking_kaggle_submission <- final_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_class) %>% 
  rename(type = .pred_class) 

vroom_write(x=stacking_kaggle_submission, file="./Submissions/StackedPreds8.csv", delim=",")

##Preds 1: NB, Boosted, RT
##Preds 2: NB, Boosted, RT, RF
##Preds 3: NB, Boosted, RF
##Preds 4: NB, Boosted
##Preds 5: NB, Radial SVM
##Preds 6: NB, Boosted, Radial SVM
##Preds 7: Boosted, Radial SVM
##Preds 8: Multi, NB

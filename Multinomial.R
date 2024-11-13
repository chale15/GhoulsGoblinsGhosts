library(kernlab)
library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(embed)
library(naivebayes)
library(discrim)
library(nnet)
library(glmnet)

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/ggg_kaggle")

train <- vroom("train.csv")
test <- vroom("test.csv")

my_recipe_multi <- recipe(type~., data = train) %>% 
  step_mutate_at(color, fn = factor) %>% 
  step_mutate(bone_flesh = bone_length * rotting_flesh,
              bone_hair = bone_length * hair_length,
              bone_soul = bone_length * has_soul,
              flesh_hair = rotting_flesh * hair_length,
              flesh_soul = rotting_flesh * has_soul,
              hair_soul = hair_length * has_soul) %>% 
  step_mutate(id, feature = id) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>% 
  step_smote(all_outcomes(), neighbors=3) %>% 
  step_range(all_numeric_predictors(), min=0, max=1)

## Multinomial Model
multi_model <- multinom_reg(penalty=tune(), mixture = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("glmnet")

# multi_model <- multinom_reg(penalty=penalty_mean, mixture = mixture_mean) %>% 
#   set_mode("classification") %>% 
#   set_engine("glmnet")

multi_wf <- workflow() %>% 
  add_model(multi_model) %>% 
  add_recipe(my_recipe_multi)

folds <- vfold_cv(train, v = 10, repeats=1)

tuning_grid_multi <- grid_regular(penalty(), mixture(), levels = c(20, 20))

multi_models <- multi_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid_multi, 
            metrics = metric_set(roc_auc, accuracy))

# best_tunes_multi <- multi_models %>% show_best(metric='roc_auc', n=10)
# mixture_mean <- mean(best_tunes_multi$mixture)
# penalty_mean <- mean(best_tunes_multi$penalty)

best_tune_multi <- multi_models %>% select_best(metric='accuracy')

final_workflow_multi <- multi_wf %>% 
  finalize_workflow(best_tune_multi) %>% 
  fit(data = train)

multi_preds <- predict(final_workflow_multi, 
                            new_data = test,
                            type = 'class')

multi_submission <- multi_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_class) %>% 
  rename(type = .pred_class) 


vroom_write(x=multi_submission, file="./Submissions/MultiPreds17.csv", delim=",")


library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(embed)
library(naivebayes)
library(discrim)
library(keras)

#Read Data

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/ggg_kaggle")

train <- vroom("train.csv")
test <- vroom("test.csv")

my_recipe <- recipe(type~., data = train) %>% 
  step_mutate_at(color, fn = factor) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_range(all_numeric_predictors(), min=0, max=1)


#Neural Net Model

nn_model <- mlp(hidden_units=tune(),
                epochs=50,
                activation = 'softmax') %>% 
  set_mode("classification") %>% 
  set_engine("keras")

nn_workflow <- workflow() %>% 
  add_model(nn_model) %>% 
  add_recipe(my_recipe)

tuning_grid <- grid_regular(hidden_units(range=c(1, 20)), levels = 20)

folds <- vfold_cv(train, v = 10, repeats=1)

cv_results <- nn_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid, 
            metrics = metric_set(accuracy, roc_auc))

best_tune <- cv_results %>% select_best(metric='accuracy')

final_workflow <- nn_workflow %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train)

nn_preds <- predict(final_workflow, 
                    new_data = test,
                    type = 'class')

nn_submission <- nn_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_class) %>% 
  rename(type = .pred_class) 


vroom_write(x=nn_submission, file="./Submissions/NNPreds1.csv", delim=",")

cv_results %>%
  collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  ggplot(aes(x = hidden_units, y = mean)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  labs(
    title = "Cross-Validation Results for Neural Network Model",
    x = "Number of Hidden Units",
    y = "Mean Accuracy"
  ) +
  theme_minimal()

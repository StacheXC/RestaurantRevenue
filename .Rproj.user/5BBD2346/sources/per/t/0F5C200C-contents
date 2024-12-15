library(vroom)
library(tidymodels)

train = vroom("train.csv")
test = vroom("test.csv")

my_recipe = recipe(revenue ~ ., train) |> 
  step_rename(Open_Date = `Open Date`, City_Group = `City Group`) |> 
  step_mutate(Open_Date = as.Date(Open_Date, format = "%m/%d/%Y")) |> 
  step_date(Open_Date, features = c("year", "month", "doy", "dow")) |> 
  step_other(all_nominal_predictors(), threshold = 0.05) |> 
  step_rm(c(Id, Open_Date))

my_model = rand_forest(mtry = tune(),
                       trees = 500,
                       min_n = tune()) |> 
  set_engine("ranger") |> 
  set_mode("regression")

wf = workflow() |> 
  add_recipe(my_recipe) |> 
  add_model(my_model)

tuning_grid = grid_regular(mtry(range = c(1, 44)),
                           min_n(),
                           levels = 5)

folds = vfold_cv(train, v = 10, repeats = 1)

CV_results = wf |> 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse),
            control=control_grid(verbose=TRUE))

bestTune = CV_results |> 
  select_best(metric = "rmse")

final_wf = wf |> 
  finalize_workflow(bestTune) |> 
  fit(data = train)

preds = predict(final_wf, new_data = test)

kaggle_submission = preds |> 
  bind_cols(test) |> 
  select(Id, .pred) |> 
  rename(Prediction=.pred)

vroom_write(x = kaggle_submission,
            file = "./forestpreds2.csv",
            delim = ",")
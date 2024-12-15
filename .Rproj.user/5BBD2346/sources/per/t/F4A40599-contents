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

my_model = bart(trees = 50) |> 
  set_engine("dbarts") |> 
  set_mode("regression")

wf = workflow() |> 
  add_recipe(my_recipe) |> 
  add_model(my_model) |> 
  fit(train)

preds = predict(wf, new_data = test)

kaggle_submission = preds |> 
  bind_cols(test) |> 
  select(Id, .pred) |> 
  rename(Prediction=.pred)

vroom_write(x = kaggle_submission,
            file = "./bartpreds.csv",
            delim = ",")

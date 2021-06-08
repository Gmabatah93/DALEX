library(tidyverse)
library(DALEX)
theme_set(theme_minimal())

# TITANIC: Data ----

# Data
titanic <- DALEX::titanic_imputed %>% as_tibble()

# EDA
titanic %>% skimr::skim()

titanic %>% 
  ggplot(aes(age)) +
  geom_histogram(bins = 30)
titanic %>% 
  ggplot(aes(fare)) +
  geom_histogram(bins = 30)
titanic %>% 
  ggplot(aes(gender, fill = factor(survived))) +
  geom_bar(position = "fill")


# MODELING: Fit
tit_mod_log <- glm(survived ~ gender + age + class + sibsp + parch + fare + embarked,
                   data = titanic, family = "binomial")
tit_mod_rf <- ranger::ranger(survived ~ gender + age + class + sibsp + parch + fare + embarked,
                             data = titanic)
tit_mod_gbm <- gbm::gbm(survived ~ gender + age + class + sibsp + parch + fare + embarked,
                        data = titanic,
                        n.trees = 10000, distribution = "bernoulli")
tit_mod_svm <- e1071::svm(survived ~ gender + age + class + sibsp + parch + fare + embarked,
                          data = titanic, type = "C-classification", probability = TRUE)  

# PREDICTIONS: 
# - Johnny D
johnny_d <- tibble(
  class = factor("1st", 
                 levels = c("1st", "2nd", "3rd", 
                            "deck crew", "engineering crew", 
                           "restaurant staff", "victualling crew")),
  gender = factor("male", 
                  levels = c("female", "male")),
  age = 8, sibsp = 0, parch = 0, fare = 72,
  embarked = factor("Southampton", 
                    levels = c("Belfast","Cherbourg","Queenstown","Southampton")))
# - Henry
henry <- tibble(
  class = factor("1st", 
                 levels = c("1st", "2nd", "3rd", 
                            "deck crew", "engineering crew", 
                            "restaurant staff", "victualling crew")),
  gender = factor("male", 
                  levels = c("female", "male")),
  age = 47, sibsp = 0, parch = 0, fare = 25,
  embarked = factor("Cherbourg",
                    levels = c("Belfast","Cherbourg","Queenstown","Southampton")))

# - Logistic Regression
pred_log <- tit_mod_log %>% predict(johnny_d, type = "response")
# - Random Forrest
pred_rf <- predict(tit_mod_rf, johnny_d, type = "response")$predictions
# - GBM
pred_gbm <- tit_mod_gbm %>% predict(johnny_d, type = "response", n.trees = 10000)
# - svm
pred_svm <- predict(tit_mod_svm, johnny_d, probability = TRUE)



# MODEL: Explainer
tit_exp_log <- explain(model = tit_mod_log,
                       data = titanic[,-8],
                       y = titanic$survived,
                       label = "Logistic Regression",
                       type = "classification")
tit_exp_rf <- explain(model = tit_mod_rf,
                      data = titanic[,-8],
                      y = titanic$survived,
                      label = "Random Forrest",
                      type = "classification")
tit_exp_gbm <- explain(model = tit_mod_gbm,
                       data = titanic[,-8],
                       y = titanic$survived == "yes",
                       label = "Gradient Boosting Machine",
                       type = "classification")
tit_exp_svm <- explain(model = tit_mod_svm,
                       data = titanic[,-8],
                       y = titanic$survived == "yes",
                       label = "Support Vector Machine",
                       type = "classification")

# TITANIC: Instance-Level ----

# Break-Down: 
# - "Break-Down"
tit_BD_rf_Henry <- predict_parts(explainer = tit_exp_rf,
                                 new_observation = henry,
                                 type = "break_down")
tit_BD_rf_Henry %>% plot()

tit_BD_rf_Henry_ordered <- predict_parts(explainer = tit_exp_rf,
                                         new_observation = henry,
                                         type = "break_down",
                                         order = c("class","age","gender","fare",
                                                   "parch","sibsp","embarked"),
                                         keep_distributions = TRUE)
tit_BD_rf_Henry_ordered %>% plot(max_features = 3, plot_distributions = TRUE)

tit_iBD_rf_Henry <- predict_parts(explainer = tit_exp_rf,
                                  new_observation = henry,
                                  type = "break_down_interactions")
tit_iBD_rf_Henry %>% plot()

# - "Shapley Values" 
tit_shap_rf_Henry <- predict_parts(explainer = tit_exp_rf,
                                   new_observation = henry,
                                   type = "shap",
                                   B = 25)
tit_shap_rf_Henry %>% plot(show_boxplots = FALSE)

# - "Local Interpretable Model-Agnostic Explanations (LIME)"
library(DALEXtra)
library(lime)
library(localModel)
library(iml)

model_type.dalex_explainer <- DALEXtra::model_type.dalex_explainer
predict_model.dalex_explainer <- DALEXtra::predict_model.dalex_explainer

lime_Johnny <- predict_surrogate(explainer = tit_exp_log,
                                 new_observation = johnny_d,
                                 n_features = 3,
                                 n_permutations = 1000,
                                 type = "lime")
lime_Johnny %>% plot()

locMod_Johnny <- predict_surrogate(explainer = tit_exp_log,
                                   new_observation = johnny_d,
                                   size = 1000,
                                   seed = 1,
                                   type = "localModel")

iml_Johnny <- predict_surrogate(explainer = tit_exp_log,
                                new_observation = johnny_d,
                                k = 3,
                                type = "iml")

iml_Johnny %>% plot()








# Ceteris-paribus Profiles
tit_cp_rf <- predict_profile(explainer = tit_exp_rf,
                             new_observation = johnny_d)
tit_cp_log <- predict_profile(explainer = tit_exp_log,
                              new_observation = johnny_d)

tit_cp_log %>% plot(variables = c("age","fare"))

# - Oscillations
tit_osc_log_Henry_Uni <- predict_parts(explainer = tit_exp_log,
                                       new_observation = henry,
                                       type = "oscillations_uni")



tit_osc_log_Henry %>% plot()
tit_osc_rf_Henry %>% plot()
tit_osc_gbm_Henry %>% plot()
tit_osc_svm_Henry %>% plot()

predict_parts(explainer = tit_exp_rf,
              new_observation = Johnny_D,
              variable_splits = list(age = seq(0,65,0.1),
                                     fare = seq(0,200,0.1),
                                     gender = unique(tit$gender),
                                     class = unique(tit$class)),
              type = "oscillations") %>% plot()


# INSTANCE Evaluation: What if Analysis
tit_cp_log_HENRY <- predict_profile(explainer = tit_exp_log,
                                    new_observation = Henry_E)
tit_cp_rf_HENRY <- predict_profile(explainer = tit_exp_rf,
                                   new_observation = Henry_E)
tit_cp_gbm_HENRY <- predict_profile(explainer = tit_exp_gbm,
                                    new_observation = Henry_E)
tit_cp_svm_HENRY <- predict_profile(explainer = tit_exp_svm,
                                    new_observation = Henry_E)

plot(tit_cp_log_HENRY, tit_cp_rf_HENRY, tit_cp_gbm_HENRY, tit_cp_svm_HENRY,
     variables = c("age","fare"), color = "_label_")
plot(tit_cp_log_HENRY, tit_cp_rf_HENRY, tit_cp_gbm_HENRY, tit_cp_svm_HENRY,
     variables = c("gender","class","embarked"), variable_type = "categorical",
     categorical_type = "bars")

# Local-Diagnostics
ld_rf <- predict_diagnostics(explainer = tit_exp_rf,
                             new_observation = Henry_E,
                             neighbors = 100)
ld_rf_age <- predict_diagnostics(explainer = tit_exp_rf,
                                 new_observation = Henry_E,
                                 neighbors = 10,
                                 variables = "age")
ld_rf_class <- predict_diagnostics(explainer = tit_exp_rf,
                                   new_observation = Henry_E,
                                   neighbors = 10,
                                   variables = "class")
ld_rf %>% plot()
ld_rf_age %>% plot()
ld_rf_class %>% plot()


# TITANIC: Dataset-Level ----

# MODEL Evaluation: Performance
tit_pref_log <- model_performance(explainer = tit_exp_log)
tit_pref_rf <- model_performance(explainer = tit_exp_rf)
tit_pref_gbm <- model_performance(explainer = tit_exp_gbm)
tit_pref_svm <- model_performance(explainer = tit_exp_svm)

plot(tit_pref_log, tit_pref_rf, tit_pref_gbm, tit_pref_svm,
     geom = "roc")
plot(tit_pref_log, tit_pref_rf, tit_pref_gbm, tit_pref_svm,
     geom = "lift")

# MODEL Evaluation: Variable Importance
loss_accuracy(observed = titanic_imputed$survived,
              predicted = predict(tit_mod_log, tit))
loss_one_minus_auc(observed = titanic_imputed$survived,
                   predicted = predict(tit_mod_log, tit))

tit_vip_log <- model_parts(explainer = tit_exp_log,  
                           type = "difference")
tit_vip_rf <- model_parts(explainer = tit_exp_rf,  
                          type = "difference")
tit_vip_gbm <- model_parts(explainer = tit_exp_gbm,  
                           type = "difference")
tit_vip_svm <- model_parts(explainer = tit_exp_svm,  
                           type = "difference")
plot(tit_vip_log, tit_vip_rf, tit_vip_gbm, tit_vip_svm)

# MODEL Evaluation: Partial Dependence
tit_pdp_rf <- model_profile(explainer = tit_exp_rf,
                            variables = "age")

tit_pdp_rf %>% plot(geom = "profiles")
# - Clustered
tit_pdp_clus_rf <- model_profile(explainer = tit_exp_rf,
                                 variables = "age",
                                 k = 3)
tit_pdp_clus_rf %>% plot(geom = "profiles")
# - Group
tit_pdp_group_rf <- model_profile(explainer = tit_exp_rf,
                                  variables = "age",
                                  groups = "gender")
tit_pdp_group_rf %>% plot(geom = "profiles")
# - Contrastive
tit_pdp_log <- model_profile(explainer = tit_exp_log,
                             variables = "age")
plot(tit_pdp_rf, tit_pdp_log)




# Local-dependence and Accumulated-local Profiles

# Residual Diagnostic


# APARTMENTS: DATA ----

# DATA
apt <- DALEX::apartments %>% as_tibble()
# EDA 
# - summary
apt %>% skimr::skim()
# - graph
apt %>% 
  ggplot(aes(m2.price)) +
  geom_histogram(bins = 30)
apt %>% 
  ggplot(aes(construction.year, m2.price)) +
  geom_point() +
  geom_smooth()
apt %>% 
  ggplot(aes(surface, m2.price)) +
  geom_point() +
  geom_smooth()

# MODEL: Fit  
apt_mod_lm <- lm(m2.price ~ ., data = apt)
apt_mod_rf <- ranger::ranger(m2.price ~ ., data = apt)  
apt_mod_gbm <- gbm::gbm(m2.price ~ ., data = apt)
apt_mod_svm <- e1071::svm(m2.price ~ ., data = apt)

# MODEL: Predictions
pred_apt_lm <- predict(apt_mod_lm, newdata = apartments_test[1:6,])
pred_apt_rf <- predict(apt_mod_rf, data = apartments_test[1:6,])$predictions
pred_apt_gbm <-predict(apt_mod_gbm, apartments_test[1:6,], type = "response")
pred_apt_svm <- predict(apt_mod_svm, apartments_test[1:6,], probability = TRUE)

tibble(
  Actual = apartments_test$m2.price[1:6],
  lm = pred_apt_lm,
  rf = pred_apt_rf,
  gbm = pred_apt_gbm,
  svm = pred_apt_svm
)

# MODEL: Explainers
apt_exp_lm <- explain(model = apt_mod_lm,
                      data = apartments_test[,-1],
                      y = apartments_test$m2.price,
                      label = "Linear Regression",
                      type = "regression")
apt_exp_rf <- explain(model = apt_mod_rf,
                      data = apartments_test[,-1],
                      y = apartments_test$m2.price,
                      label = "Random Forrest",
                      type = "regression")
apt_exp_rf <- explain(model = apt_mod_gbm,
                      data = apartments_test[,-1],
                      y = apartments_test$m2.price,
                      label = "Random Forrest",
                      type = "regression")
apt_exp_svm <- explain(model = apt_mod_svm,
                       data = apartments_test[,-1],
                       y = apartments_test$m2.price,
                       label = "Support Vector Machine",
                       type = "regression")


# APARTMENT: Instance-Level ----

# APRATMENT: Dataset-Level ----

# Model-Performance
apt_eval_rf <- model_performance(explainer = apt_exp_rf)
apt_eval_lm <- model_performance(explainer = apt_exp_lm)

plot(apt_eval_rf, apt_eval_lm, geom = "histogram")
plot(apt_eval_rf, geom = "prc")

# Variable Importance Measures
apt_vip_rf <- model_parts(explainer = apt_exp_rf,
                          loss_function = loss_root_mean_square,
                          B = 50, 
                          type = "difference")

apt_vip_lm <- model_parts(explainer = apt_exp_lm,
                          loss_function = loss_root_mean_square,
                          B = 50, 
                          type = "difference")

apt_vip_svm <- model_parts(explainer = apt_exp_svm,
                           loss_function = loss_root_mean_square,
                           B = 50, 
                           type = "difference")


plot(apt_vip_rf, apt_vip_lm, apt_vip_svm)


# Partial Dependence Profile

# Residual Diagnostic
apt_mr_lm <- model_performance(explainer = apt_exp_lm)
apt_mr_rf <- model_performance(explainer = apt_exp_rf)

plot(apt_mr_lm, apt_mr_rf, geom = "histogram")
plot(apt_mr_lm, apt_mr_rf, geom = "boxplot")

apt_md_lm <- model_diagnostics(explainer = apt_exp_lm)
apt_md_rf <- model_diagnostics(explainer = apt_exp_rf)

plot(apt_md_rf, variable = "y", yvariable = "residuals")
plot(apt_md_rf, variable = "y", yvariable = "y_hat")
plot(apt_md_rf, variable = "ids", yvariable = "residuals")
plot(apt_md_rf, variable = "y_hat", yvariable = "abs_residuals")

#
# FIFA: DAtA ----

# EDA
# - summary
fifa %>% skimr::skim()
# - target
fifa %>% 
  ggplot(aes(log10(value_eur))) +
  geom_histogram(bins = 100)
fifa %>% 
  ggplot(aes(age, log10(value_eur))) +
  geom_point() +
  geom_smooth()
fifa %>% 
  ggplot(aes(skill_ball_control, log10(value_eur))) +
  geom_point() +
  geom_smooth()
fifa %>% 
  ggplot(aes(skill_dribbling, log10(value_eur))) +
  geom_point() +
  geom_smooth()
fifa %>% 
  ggplot(aes(movement_reactions, log10(value_eur))) +
  geom_point() +
  geom_smooth()

# MODELING
# - data
fifa$value_eur_Log <- log10(fifa$value_eur)
fifa_small <- fifa %>% 
  select(-c("value_eur","wage_eur","overall","potential"))
# - fit
fifa_gbm_deep <- gbm::gbm(value_eur_Log ~ ., data = fifa_small,
                          n.trees = 250, 
                          interaction.depth = 4, 
                          distribution = "gaussian")
# - explainer 
fifa_gbm_exp_deep <- explain(fifa_gbm_deep, 
                            data = fifa_small, 
                            y = 10^fifa_small$value_eur_Log, 
                            predict_function = function(m,x) 10^predict(m, x, n.trees = 250),
                            label = "GBM deep")
# FIFA: Dataset level ----

# Model-Performance
fifa_mp_deep <- model_performance(explainer = fifa_gbm_exp_deep)
fifa_md_deep <- model_diagnostics(explainer = fifa_gbm_exp_deep)
fifa_md_deep %>% 
  plot(variable = "y", yvariable = "y_hat") +
  scale_x_continuous("Value in Euro", trans = "log10") +
  scale_y_continuous("Predicted value in Euro", trans = "log10") +
  geom_abline(slope = 1)

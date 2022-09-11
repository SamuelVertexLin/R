rm(list = ls())

#--------------------------------------------- data preparation

library(data.table)

load("D:/Machine Learning & Financial Econometrics/data_ml.RData")

data_ml = as.data.table(data_ml)
train_data <- data_ml[ date %between% c("2006-01-01","2007-12-31")]
valid_data <- data_ml[ date %between% c("2008-01-01","2008-12-31")]
test_data <- data_ml[ date %between% c("2009-01-01","2009-12-31")]

names(train_data)

# select parameters, with date on the first row and forcasting R12M_Usd in the end

train_data <- train_data[, c("date", "Mkt_Cap_12M_Usd", "Pb", "Sales_Ps", "Mom_11M_Usd",
                             "Vol1Y_Usd", "Roa", "Mom_Sharp_11M_Usd", "Ebit_Noa", "Roe", "Share_Turn_12M",
                             "Ev_Ebitda", "Ebitda_Margin", "Asset_Turnover", "Capex_Sales", "Total_Debt_Capital",
                             "Op_Prt_Margin", "R12M_Usd")]

valid_data <- valid_data[, c("date", "Mkt_Cap_12M_Usd", "Pb", "Sales_Ps", "Mom_11M_Usd",
                             "Vol1Y_Usd", "Roa", "Mom_Sharp_11M_Usd", "Ebit_Noa", "Roe", "Share_Turn_12M",
                             "Ev_Ebitda", "Ebitda_Margin", "Asset_Turnover", "Capex_Sales", "Total_Debt_Capital",
                             "Op_Prt_Margin", "R12M_Usd")]

test_data <- test_data[, c("date", "Mkt_Cap_12M_Usd", "Pb", "Sales_Ps", "Mom_11M_Usd",
                           "Vol1Y_Usd", "Roa", "Mom_Sharp_11M_Usd", "Ebit_Noa", "Roe", "Share_Turn_12M",
                           "Ev_Ebitda", "Ebitda_Margin", "Asset_Turnover", "Capex_Sales", "Total_Debt_Capital",
                           "Op_Prt_Margin", "R12M_Usd")]
names(train_data)

# obtain only December

library(lubridate)
class(train_data$date)

train_data <- train_data[month(train_data$date) == 12, ]
valid_data <- valid_data[month(valid_data$date) == 12, ]
test_data <- test_data[month(test_data$date) == 12, ]

train_data <- na.omit(train_data)
valid_data <- na.omit(valid_data)
test_data <- na.omit(test_data)

train_data$date
valid_data$date
test_data$date

#-------------1. Ridge regression and lasso

library(ISLR2)
library(gglasso)
library(splines)

# create x, y

train_data1 <- train_data[, c("Mkt_Cap_12M_Usd", "Pb", "Sales_Ps", "Mom_11M_Usd",
                              "Vol1Y_Usd", "Roa", "Mom_Sharp_11M_Usd", "Ebit_Noa", "Roe", "Share_Turn_12M",
                              "Ev_Ebitda", "Ebitda_Margin", "Asset_Turnover", "Capex_Sales", "Total_Debt_Capital",
                              "Op_Prt_Margin")]
train_data2 <- train_data$R12M_Usd

# convert into matrix

class(train_data1)
train_data1 <- as.matrix(train_data1)
is.matrix((train_data1))

#-----------Ridge regression
library(glmnet)

lambda_grid <- 10^seq(3, -3, length = 100)
round(lambda_grid, digits = 4)

ridge_mod <- glmnet(train_data1, train_data2, alpha = 0, lambda = lambda_grid)
ridge_mod
plot(ridge_mod)

#-----------Lasso

lasso_mdl <- glmnet(train_data1, train_data2, alpha = 1, lambda = lambda_grid)
lasso_mdl
plot(lasso_mdl)

#------(b).

# create x, y

valid_data1 <- valid_data[, c("Mkt_Cap_12M_Usd", "Pb", "Sales_Ps", "Mom_11M_Usd",
                              "Vol1Y_Usd", "Roa", "Mom_Sharp_11M_Usd", "Ebit_Noa", "Roe", "Share_Turn_12M",
                              "Ev_Ebitda", "Ebitda_Margin", "Asset_Turnover", "Capex_Sales", "Total_Debt_Capital",
                              "Op_Prt_Margin")]
valid_data2 <- valid_data$R12M_Usd

# convert into matrix

is.matrix((valid_data1))
valid_data1 <- as.matrix(valid_data1)
is.matrix((valid_data1))

#  choose the tuning parameter (lambda) which gives the lowest MSE (Both Ridge regression & Lasso)

#----Ridge regression

y_val_hat1 = predict(ridge_mod, newx=valid_data1) # predict 2008/12 R12M
y_val_hat1

val_error1 = colMeans((valid_data2 - y_val_hat1)^2)
round(val_error1, 5)
plot(val_error1)

minMSE_lambda1 = lambda_grid[which.min(val_error1)] # minimum MSE's lambda
minMSE_lambda1

#----Lasso

y_val_hat2 = predict(lasso_mdl, newx=valid_data1) # predict 2008/12 R12M
y_val_hat2

val_error2 = colMeans((valid_data2 - y_val_hat2)^2)
round(val_error2, 5)
plot(val_error2)

minMSE_lambda2 = lambda_grid[which.min(val_error2)] # minimum MSE's lambda
minMSE_lambda2

# Ridge regression & Lasso minMSE_lambda are not the same

#-------(c).

#(只留下每年12月的)

full_data <- rbind(train_data, valid_data)
full_data

# create x, y

full_data1 <- full_data[, c("Mkt_Cap_12M_Usd", "Pb", "Sales_Ps", "Mom_11M_Usd",
                            "Vol1Y_Usd", "Roa", "Mom_Sharp_11M_Usd", "Ebit_Noa", "Roe", "Share_Turn_12M",
                            "Ev_Ebitda", "Ebitda_Margin", "Asset_Turnover", "Capex_Sales", "Total_Debt_Capital",
                            "Op_Prt_Margin")]
summary(full_data1)
full_data2 <- full_data$R12M_Usd

# convert into matrix

full_data1 <- as.matrix(full_data1)
is.matrix((full_data1))

# Re-estimate the model with sample from 2006~2008 and minMSE_lambda (Both Ridge regression & Lasso)

#------Ridge regression

ridge_mod2 = glmnet(full_data1, full_data2, alpha=0,lambda=minMSE_lambda1)
ridge_mod2

#------Lasso

lasso_mdl2 = glmnet(full_data1, full_data2, alpha=1,lambda=minMSE_lambda2)
lasso_mdl2

#-------(d).

# prediction error of Ridge regression & Lasso

pred_data <- test_data

# create x, y

pred_data1 <- pred_data[, c("Mkt_Cap_12M_Usd", "Pb", "Sales_Ps", "Mom_11M_Usd",
                            "Vol1Y_Usd", "Roa", "Mom_Sharp_11M_Usd", "Ebit_Noa", "Roe", "Share_Turn_12M",
                            "Ev_Ebitda", "Ebitda_Margin", "Asset_Turnover", "Capex_Sales", "Total_Debt_Capital",
                            "Op_Prt_Margin")]
pred_data1 <- as.matrix(pred_data1)
is.matrix(pred_data1)
pred_data2 <- pred_data$R12M_Usd

#------Ridge Regression

y_forecast1 = predict(ridge_mod2 ,newx = pred_data1, type="response")
round(y_forecast1, 4)

pred_error1 = pred_data2 - y_forecast1
round(pred_error1, 4)

ridge_RMSE = sqrt(mean(pred_error1^2))
ridge_RMSE

ridge_PCSP = mean(sign(pred_data2) == sign(y_forecast1)) # percentage of correct sign prediction
ridge_PCSP

ridge_RC = cor(pred_data2, y_forecast1 ,method="spearman")
ridge_RC

#------Lasso

y_forecast2 = predict(lasso_mdl2 ,newx = pred_data1, type="response")
round(y_forecast2, 4)

pred_error2 = pred_data2 - y_forecast2
round(pred_error2, 4)

lasso_RMSE = sqrt(mean(pred_error2^2))
lasso_RMSE

lasso_PCSP = mean(sign(pred_data2) == sign(y_forecast2)) # percentage of correct sign prediction
lasso_PCSP

lasso_RC = cor(pred_data2, y_forecast2 ,method="spearman")
lasso_RC

#-------------2.

#------(a).

### Run principal component regression
library(pls)

###
set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(pred_data), replace = TRUE)
test <- (!train)

x = c("Mkt_Cap_12M_Usd","Pb","Sales_Ps","Mom_11M_Usd","Vol1Y_Usd","Roa",
      "Mom_Sharp_11M_Usd","Ebit_Noa","Roe","Share_Turn_12M","Ev_Ebitda",
      "Ebitda_Margin","Asset_Turnover","Capex_Sales","Total_Debt_Capital",
      "Op_Prt_Margin")
k <- pred_data1
y <- pred_data2
y.test <- y[test]
  
pcr.fit <- pcr(as.formula(paste("R12M_Usd ~ ", paste(x,collapse="+"))), 
               data = pred_data, 
               subset = train,
               scale = TRUE, 
               validation = "CV")
as.formula(paste("R12M_Usd ~ ", paste(x,collapse="+")))
summary(pcr.fit)
validationplot(pcr.fit, val.type = "MSEP")

### Restrict to 5 components

pcr.fit <- pcr(as.formula(paste("R12M_Usd ~ ", paste(x,collapse="+"))), 
               data = pred_data, 
               scale = TRUE, 
               ncomp = 5)
summary(pcr.fit)
validationplot(pcr.fit, val.type = "MSEP")

### Partial Least Squares

### set.seed(1)

pls.fit <- plsr(as.formula(paste("R12M_Usd ~ ", paste(x,collapse="+"))),
                data = pred_data, 
                subset = train, 
                scale = TRUE, 
                validation = "CV")
summary(pls.fit)
validationplot(pls.fit, val.type = "MSEP")

###
pls.pred <- predict(pls.fit, k[test, ], ncomp = 5)
mean((pls.pred - y.test)^2)
###

pls.fit <- plsr(as.formula(paste("R12M_Usd ~ ", paste(x,collapse="+"))), data = pred_data, scale = TRUE,
                ncomp = 5)
summary(pls.fit)
###

#-------------3.

#------(a).

library(ANN2)
library(ISLR2)

# remove the obs. with missing values

set.seed(1)
x <- pred_data1 # 2009 16 parameters
y <- pred_data2 # 2009 R12MUsd

for (z in 0:3){
  NN <- neuralnetwork(x, y, 
                    hidden.layers = c(64,32,16),  
                    optim.type = "sgd",
                    loss.type = "squared",
                    activ.functions = "relu",
                    regression = T,  
                    batch.size = 32*(2^z),  #Please consider 32, 64, 128, and 256.
                    n.epochs = 30,
                    drop.last = T,
                    val.prop = 0.1,
                    random.seed = 1)
  
  yhat = predict(NN, newdata = x)$predictions
  result <-mean((y - yhat)^2)
  cat("When batch size = :", 32*(2^z), "Mean((y - yhat)^2) :", result)
cat(step = "\n")
}

#----------4.
library(xgboost)

Data_train <- xgb.DMatrix(data = pred_data1, label = pred_data2) 

set.seed(1)
for (x in 1:6){
  param <- list(max_depth = x, eta = 0.01,  # Please consider 1, 2, 3, 4, 5, and 6.
              objective = "reg:squarederror",
              colsample_bytree = 0.3)
  boosted_tree <- xgb.train(param, Data_train, nrounds = 100)
  yhat <- predict(boosted_tree, pred_data1)
  result <- mean((yhat - pred_data2)^2)
  cat("When max_depth :",  x, "Mean :", result)
cat(step = "\n")
}
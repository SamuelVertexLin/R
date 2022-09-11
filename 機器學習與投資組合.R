rm(list = ls())
options(max.print = .Machine$integer.max)

#
#------- Data Preparation -------# 
library(data.table)
load("D:/Machine Learning & Financial Econometrics/data_ml.RData")
data_ml <- as.data.table(data_ml)
data_ml <- data_ml[data_ml$stock_id %in% c(567, 737, 829, 391, 1013), ]
data_ml$stock_id

#
#------- Portfolio Construction -------# 
library(lubridate)
library(CVXR)
library(matrixStats)
library(xgboost)

#
#--- (a) 1/N Portfolio ---#
#

# Subset Data
data_ml_a <- data_ml[date %between% c("2018-01-31","2019-01-31")]
data_ml_a <- data_ml_a[order(data_ml_a$date), ]
data_ml_a <- dcast(data_ml_a, date~stock_id, value.var="R1M_Usd") # "setkey()" will be a better program syntax #

# Monthly Individual Stock Returns
stock_ret_a <- as.matrix(data_ml_a[,-1])

# Construct Weight
weight_a <- matrix(0.2, nrow = 5, ncol = 1)
weight_a

# Time Series Of Portfolio Returns
portf_ret_a = stock_ret_a %*% weight_a

# Portfolio Concentration Measure
portf_concentr_a = sum((weight_a)**2)
portf_concentr_a 
# 0.2

# Annualized Mean Returns
ann_mean_ret_a = 12*mean(portf_ret_a)
ann_mean_ret_a 
# 0.001292308

# Annualized Volatility
ann_vol_a = (12**0.5) * sd(portf_ret_a)
ann_vol_a 
# 0.2117723

# Mean-Volatility Ratio
mean_vol_rat_a = ann_mean_ret_a / ann_vol_a
mean_vol_rat_a 
# 0.006102346

#
#--- (b) Inverse-Volatility-Weighted Portfolio ---#
#

# Subset Data
data_ml_b <- data_ml[ date %between% c("1998-11-30","2017-12-31")]
data_ml_b <- data_ml_b[order(data_ml_b$date), ]
data_ml_b <- dcast(data_ml_b, date~stock_id, value.var="R1M_Usd")

# Monthly Individual Stock Returns
stock_ret_b = stock_ret_a

# Construct Standard Deviation & Weight
st_dev_b <- colSds(as.matrix(data_ml_b[,-1]))

weight_b = (1/st_dev_b)/sum(1/st_dev_b)
weight_b 
# 0.2450360 0.2172820 0.0998000 0.1247175 0.3131645

# Time Series Of Portfolio Returns
portf_ret_b = stock_ret_b %*% weight_b

# Portfolio Concentration Measure
portf_concentr_b = sum((weight_b)**2)
portf_concentr_b 
# 0.2308406

# Annualized Mean Return
ann_mean_ret_b = 12*mean(portf_ret_b)
ann_mean_ret_b 
# 0.009517894

# Annualized Volatility
ann_vol_b = (12**0.5) * sd(portf_ret_b)
ann_vol_b 
# 0.2016095

# Mean-Volatility Ratio
mean_vol_rat_b = ann_mean_ret_b / ann_vol_b
mean_vol_rat_b 
# 0.04720954

# Question
mean_vol_rat_tot <- c(mean_vol_rat_b, mean_vol_rat_a)
mean_vol_rat_max <- max(mean_vol_rat_tot)

cat ("Question: Does the inverse-volatility-weighted portfolio deliver better performance than the 1/N portfolio?")
if (mean_vol_rat_max == mean_vol_rat_a){
  cat ("Answer: Mean-Volatility Ratio: Inverse-Volatility-Weighted Portfolio < 1/N Portfolio
        The inverse-volatility-weighted portfolio doesn't deliver better performance than the 1/N portfolio")
  }else{ 
  cat ("Answer: Mean-Volatility Ratio: Inverse-Volatility-Weighted Portfolio > 1/N Portfolio
        The inverse-volatility-weighted portfolio does deliver better performance than the 1/N portfolio")
}

#
#--- (c) Mean-Variance Portfolio ---#
#

# Subset Data
data_ml_c = data_ml_b 

# Monthly Individual Stock Returns
stock_ret_c = stock_ret_a

# Construct Mean, Variance & Weight_Variable
mean_c <- colMeans(data_ml_c[,-1]) 

var_c <- cov(data_ml_c[,-1])

n_c = length(mean_c)

weight_c = Variable(n_c) 

# Required Portfolio Mean Return

portf_mean_ret_req_c = c(0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014,0.015) # mu_p

# Empty numeric vector for all 
portf_concentr_c = numeric(8)
ann_mean_ret_c = numeric(8)
ann_vol_c = numeric(8)
mean_vol_rat_c = numeric(8)

# Optimizing Weight (using CVXR)
portf_mean_ret_c = t(mean_c) %*% weight_c

portf_var_c = quad_form(weight_c, var_c)
for ( i in 1:8) {
  obj_c = 0.5 * portf_var_c
  constr_c = list(sum(weight_c) == 1, 
                       weight_c >= 0,  
               portf_mean_ret_c >= portf_mean_ret_req_c[i])
  prob_c = Problem(Minimize(obj_c), constr_c)
  result_c = solve(prob_c)
  
  # Get Optimal Weight
  weight_opt_c =  result_c$getValue(weight_c)
  
  # Time Series Of Portfolio Returns
  portf_ret_opt_c = stock_ret_c %*% weight_opt_c

  # Portfolio Concentration Measure
  portf_concentr_c[i] = sum((weight_opt_c)**2)

  # Annualized Mean Return
  ann_mean_ret_c[i] = 12*mean(portf_ret_opt_c)

  # Annualized Volatility
  ann_vol_c[i] = (12**0.5) * sd(portf_ret_opt_c)

  # Mean-Volatility Ratio
  mean_vol_rat_c[i] = ann_mean_ret_c[i] / ann_vol_c[i]
  
  # Discussion
  names(weight_opt_c) = names(mean_c)
  data.frame_c <- data.frame(weight_opt_c, row.names = names(mean_c))
  cat("\nWhen portfolio mean return =", portf_mean_ret_req_c[i],"\n")
  print(data.frame_c)
  cat("Mean-Volatility Ratio = ", mean_vol_rat_c[i],
      "\nAnnualized Mean Return = ", ann_mean_ret_c[i],
      "\nAnnualized Volatility = ", ann_vol_c[i],
      "\nMean-Volatility Ratio = ", mean_vol_rat_c[i])
  
  #Comparison (each)
  mean_vol_rat_each <- c(mean_vol_rat_c[i], mean_vol_rat_a)
  mean_vol_rat_comp <- max(mean_vol_rat_each)
  
  if (mean_vol_rat_comp == mean_vol_rat_a){
    cat ("\nThe portfolio underperforms the 1/N portfolio in terms of mean-volatility ratio.\n")
  }else{ 
    cat ("\nThe portfolio overperforms the 1/N portfolio in terms of mean-volatility ratio.\n")
  }
}

#
#--- (d) Mean-Variance Portfolio + Boosted Tree Mean Forecast ---#
#

### i - iii
# Subset Data
data_ml_d_train <- data_ml[ date %between% c("1998-11-30","2016-12-31")]
data_ml_d_train <- data_ml_d_train[order(data_ml_d_train$date), ]

# The set of predictors
X <- c("Vol1Y_Usd", "Mom_11M_Usd", "Mkt_Cap_12M_Usd")

# The sample from 1998-11-30 to 2016-12-31 for training
Xmat_train <- as.matrix(data_ml_d_train[, X, with=F])

y_train <- data_ml_d_train$R12M_Usd

# Boosted Tree Model
xgb_train_data <- xgb.DMatrix(data = Xmat_train, label = y_train) 

param <- list(max_depth = 4, 
                    eta = 0.01, 
       colsample_bytree = 1,
              objective = "reg:squarederror")  
xgb_mdl <- xgb.train(param, xgb_train_data, nrounds=100)

###iv
# The sample from 2017-12-31 for forecasting
data_ml_d_forecast <- data_ml[ date == "2017-12-31",]

Xmat_new <- as.matrix(data_ml_d_forecast[, X, with=F])

y_forecast = predict(xgb_mdl, newdata = Xmat_new)/12

# Monthly Individual Stock Returns
stock_ret_d = stock_ret_a

# Construct Mean, Variance & Weight_Variable
mean_d = y_forecast

var_d = var_c

n_d = length(y_forecast)

weight_d = Variable(n_d) 

# Required Portfolio Mean Return
portf_mean_ret_req_d = c(0.01, 0.014, 0.018, 0.022)

# Empty numeric vector for all 
portf_concentr_d = numeric(4) 
ann_mean_ret_d = numeric(4)
ann_vol_d = numeric(4)
mean_vol_rat_d = numeric(4)

# Optimizing Weight (using CVXR)
portf_mean_ret_d = t(mean_d) %*% weight_d

portf_var_d = quad_form(weight_d, var_d)

for( i in 1:4) {
  obj_d = 0.5*portf_var_d
  constr_d = list(sum(weight_d) == 1,
                       weight_d >= 0,
               portf_mean_ret_d >= portf_mean_ret_req_d[i])
  prob_d = Problem(Minimize(obj_d), constr_d) # Define the problem
  result_d = solve(prob_d)
  
  # Get Optimal Weight
  weight_opt_d =  result_d$getValue(weight_d)
  
  # Time Series Of Portfolio Returns
  portf_ret_opt_d = stock_ret_d %*% weight_opt_d
  
  # Portfolio Concentration Measure
  portf_concentr_d[i] = sum((weight_opt_d)**2)
  
  # Annualized Mean Returns
  ann_mean_ret_d[i] = 12*mean(portf_ret_opt_d)
  
  # Annualized Volatility
  ann_vol_d[i] = (12**0.5) * sd(portf_ret_opt_d)
  
  # Mean-Volatility Ratio
  mean_vol_rat_d[i] = ann_mean_ret_d[i] / ann_vol_d[i]
  
  # Discussion
  names(weight_opt_d) = names(mean_d)
  data.frame_d <- data.frame(weight_opt_d, row.names = names(mean_c)) #names(mean_c) = names(mean_d)
  cat("\nWhen portfolio mean return =", portf_mean_ret_req_d[i],"\n")
  print(data.frame_d)
  cat("Portfolio Concentration Measure = ", portf_concentr_d[i],
      "\nAnnualized Mean Return = ", ann_mean_ret_d[i],
      "\nAnnualized Volatility = ", ann_vol_d[i],
      "\nMean-Volatility Ratio = ", mean_vol_rat_d[i],"\n")
}

# Question
cat ("Question: Does the prediction based on the boosted tree improve upon the mean-variance portfolio performance?\n")
if (mean(mean_vol_rat_c) < mean(mean_vol_rat_d)){
  cat ("Answer: The boosted tree improve upon the mean-variance portfolio performance\n")
}else{ 
  cat ("Answer: The boosted tree does not improve upon the mean-variance portfolio performance\n")
}
# The boosted tree improve upon the mean-variance portfolio performance

#----- summarize the mean_vol_ratio-----#
DT = data.table(
  "1/N" = mean_vol_rat_a,
  "IVW" = mean_vol_rat_b,
  "mvp" = mean(mean_vol_rat_c),
  "xgb" = mean(mean_vol_rat_d)
)
DT
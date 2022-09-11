library(plm)
library(data.table)

load("D:/Machine Learning & Financial Econometrics/R codes for data exploration-20220322/data_ml.RData")

# http://www.mlfactor.com/data-description.html?q=data#data-description

data_ml = as.data.table(data_ml)
data_ml <- data_ml[ date %between% c("2006-12-31","2018-12-31")]

data_ml[,.N, keyby=c("date")]# report the number of firms for each month

# Run the Fama-MacBeth regression model using the estimation data
Estimation_Data = data_ml[ date %between% c("2006-12-30","2011-12-31")]
FM_reg = pmg(R12M_Usd ~ Mkt_Cap_12M_Usd + Pb + Mom_11M_Usd + Vol1Y_Usd + Roa + Op_Prt_Margin,
            data = Estimation_Data,
            index = c("date","stock_id"))
summary(FM_reg)

# Collect the parameter estimates of FM regression
B = FM_reg$coefficients

# Construct the return forecasts
Portfolio_Test_Sample = data_ml[date %between% c("2012-01-01","2019-12-31")]
Portfolio_Test_Sample[, RetForecast_FM := B[1] + B [2]*Mkt_Cap_12M_Usd + B[3]*Pb + 
                        B[4]*Mom_11M_Usd + B[5]*Vol1Y_Usd + B[6]*Roa + B[7]*Op_Prt_Margin,]

# Portfolio based on the return forecast
MarketAverage = Portfolio_Test_Sample[,.(MarketAverage = weighted.mean(R1M_Usd, w = Mkt_Cap_3M_Usd)), by=date]
Portf_FM = Portfolio_Test_Sample[,.(PReturn = weighted.mean(R1M_Usd, w = (1+RetForecast_FM)*Mkt_Cap_3M_Usd)), by=date]

Portf = merge(MarketAverage,Portf_FM)

mean(Portf[, ifelse(PReturn > MarketAverage,1,0)])
Portf[,.(ExcessMean = mean(PReturn-MarketAverage),
         Volatility = sd(PReturn),
         MarketVol = sd(MarketAverage),
         Correlation = cor(PReturn,MarketAverage))]

A = Portf[,lm(PReturn~MarketAverage)]
summary(A)

plot(Portf[,PReturn,MarketAverage])
abline(Portf[,lm(PReturn~MarketAverage)])

FFreg = lm (PReturn~MarketAverage, data = Portf)


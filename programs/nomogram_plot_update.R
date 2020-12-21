rm(list = ls())
library(rms)
library(readxl)
library(survival)
library(ResourceSelection)
library(showtext)
library(nomogramFormula)
library(devtools) 
library(rmda)
setwd("C:/projects/zhihui/14_口腔癌/口腔癌数据整理-TN-4.24/口腔癌数据整理-TN-4.24/数据挖掘更新5/回归系数表及列线图数据集")
my_data <- read_excel("seed0其他因子列线图train数据(逐步回归).xlsx")
test_data <- read_excel("seed0其他因子列线图test数据(逐步回归).xlsx")
ddist = datadist(my_data)
options(datadist='ddist')
#f = cph(Surv(times, status)~`Consuming fruit every day`+Housework
#        +`Sleeping well`+`Light diet`+Smoking+HDL,data=my_data,x=TRUE,y=TRUE,surv=TRUE)
f = cph(Surv(times, status)~.,data=my_data,x=TRUE,y=TRUE,surv=TRUE,time.inc =1095)
#validate(f, method="boot", B=1000, dxy=T)
#rcorrcens(Surv(times, status) ~ predict(f), data = my_data)
survival = Survival(f)
survival1 = function(x)survival(365,x)
survival2 = function(x)survival(1095,x)
survival3 = function(x)survival(1825,x)
nom = nomogram(f, fun = list(survival1,survival2,survival3),
               fun.at = c(.001,.01,.05,seq(.1,.9,by=.1),.95,.99,.999),
               lp=F,
               funlabel = c('1-year OS probability','3-year OS probability','5-year OS probability'))
plot(nom)
baseline.model = decision_curve(formula, data, family = binomial(link = "logit"),
                                policy = c("opt-in", "opt-out"), fitted.risk = FALSE,
                                thresholds = seq(0, 1, by = 0.01), confidence.intervals = 0.95,
                                bootstraps = 500, study.design = c("cohort", "case-control"),
                                population.prevalence)

plot_decision_curve(f)

results <- formula_rd(nomogram = nom)
points <- points_cal(formula = results$f,rd=test_data)
test_data["points"] = points
#write.table(test_data,file = '其他因子含得分test数据集.txt',sep = '\t')
 
calev3 <- calibrate(f, cmethod="KM", method="boot", u=1095, m=30, B=100,aics = 5)
calev3_plot <- plot(calev3,lwd=2,lty=1,xlab=" Predicted probability of 3-year overall survival ",ylab=list("Actual probability of 3-year overall survival"),xlim=c(0,1),ylim=c(0,1),
                    errbar.col=c(rgb(88,87,86,maxColorValue=255)),col=c(rgb(176,23,31,maxColorValue=255)),font.axis = 1,
                    font.lab = 1,family = "Times New Roman")
abline(0,1,lty=3,lwd=2,col=c(rgb(0,0,255,maxColorValue= 255)))
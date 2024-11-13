rm(list=ls())
library(glmnet)
library(ncvreg)
library(MASS)

source('Trans_Lasso.R')
source('ATL_func.R')

p <- 100
beta_t <- c(1:5,rep(0,p-5))
beta_s <- c(11,2:5,rep(0,p-5))
q <- 1

n_t <- 20
n_s <- 100

sim <- 20
beta.hat_t.non <- matrix(NA, nrow = sim, ncol = p)
beta.hat_t <- matrix(NA, nrow = sim, ncol = p)
beta.hat_t.trans <- matrix(NA, nrow = sim, ncol = p)
beta.hat_t.trans.sp <- matrix(NA, nrow = sim, ncol = p)
beta.hat_t.ind <- matrix(NA, nrow = sim, ncol = p)
beta.hat_t.adp <- matrix(NA, nrow = sim, ncol = p)

for (s in 1:sim) {
  ##################### Target
  # orthogonal
  # X_t <- mvrnorm(n_t, mu = rep(1,p), Sigma = diag(10,p)) # X~N_p(1,I)
  # partial orthogonal
  SigmaX <- matrix(NA, nrow = p, ncol = p)
  for (i in 1:p) {
    for (j in 1:p) {
      SigmaX[i,j] <- 0.05^(abs(i-j))
    }
  }
  for (k in 1:q) {
    SigmaX[k,-k] <- rep(0,p-1)
    SigmaX[-k,k] <- rep(0,p-1)
  }
  X_t <- mvrnorm(n_t, mu = rep(1,p), Sigma = SigmaX)
  e_t <- rnorm(n_t)
  Y_t <- X_t%*%beta_t + e_t
  
  ##################### Source
  # orthogonal
  # X_s <- mvrnorm(n_s, mu = rep(1,p), Sigma = diag(1,p)) # X~N_p(1,I)
  # partial orthogonal
  X_s <- mvrnorm(n_s, mu = rep(1,p), Sigma = SigmaX)
  e_s <- rnorm(n_s)
  Y_s <- X_s%*%beta_s + e_s

  ##################### Methods
  # Non-Trans Lasso
  lasso.fit_t.non <- cv.glmnet(X_t, Y_t, intercept = FALSE, nfolds = 5)
  beta.hat_t.non[s,] <- coef(lasso.fit_t.non)[-1,]
  
  # Oracle Trans-Lasso
  X <- rbind(X_t, X_s)
  y <- c(Y_t, Y_s)
  n.vec <- c(n_t, n_s)
  size.A0 <- 1
  beta.hat_t[s,] <- las.kA(X, y, A0 = 1:size.A0, n.vec = n.vec, l1=T)$beta.kA
  
  # Trans-Lasso
  prop.re1 <- Trans.lasso(X, y, n.vec, I.til = 1:5, l1 = T)
  prop.re2 <- Trans.lasso(X, y, n.vec, I.til = 6:n.vec[1], l1=T)
  beta.hat_t.trans[s,] <- (prop.re1$beta.hat + prop.re2$beta.hat) / 2

  # Trans-Lasso with a different R.hat
  prop.sp.re1 <- Trans.lasso.sp(X, y, n.vec, I.til = 1:15, l1 = T)
  prop.sp.re2 <- Trans.lasso.sp(X, y, n.vec, I.til = 16:n.vec[1], l1=T)
  beta.hat_t.trans.sp[s,] <- (prop.sp.re1$beta.sp + prop.sp.re2$beta.sp) / 2
  
  # ATL
  beta.hat_t.ind[s,] <- ATL(X_t, y_t, X_s, y_s, crossing_fitting_fold=3, threshold=c("hard"))
  beta.hat_t.adp[s,] <- ATL(X_t, y_t, X_s, y_s, crossing_fitting_fold=3, threshold=c("smooth"))
}

RMSE_nontrans <- error.hat(beta_t, beta.hat_t.non)[[1]]
RMSE_naivetrans <- error.hat(beta_t, beta.hat_t)[[1]]
RMSE_Qtrans <- error.hat(beta_t, beta.hat_t.trans)[[1]]
RMSE_Qtrans.sp <- error.hat(beta_t, beta.hat_t.trans.sp)[[1]]
RMSE_indicator <- error.hat(beta_t, beta.hat_t.ind)[[1]]
RMSE_adaptive <- error.hat(beta_t, beta.hat_t.adp)[[1]]

boxplot(RMSE_nontrans, RMSE_naivetrans, RMSE_Qtrans.sp,
        RMSE_indicator, RMSE_adaptive,
        names = c('Non-Trans Lasso', 'Oracle Trans-Lasso', 'Trans-Lasso',
                  'H-ATL', 'S-ATL'),
        ylab = 'RMSE')

summary(RMSE_nontrans)
summary(RMSE_naivetrans)
summary(RMSE_Qtrans)
summary(RMSE_Qtrans.sp)
summary(RMSE_indicator)
summary(RMSE_adaptive)

cbind(c('Non-Trans Lasso','Oracle Trans-Lasso','Trans-Lasso','Trans-Lasso.sp','H-ATL','S-ATL'),
      c(rbind(mean(RMSE_nontrans), mean(RMSE_naivetrans), mean(RMSE_Qtrans), mean(RMSE_Qtrans.sp), mean(RMSE_indicator), mean(RMSE_adaptive))))




rm(list=ls())
library(glmnet)
library(ncvreg)
library(MASS)

source('Trans_Lasso.R')


# tuning
p = 100
beta_t <- c(1:5,rep(0,p-5))
beta_s <- c(11,2:5,rep(0,p-5))
q = 1

n_t = 20
n_s = 100

gamma = 0.05

sim = 20
beta.hat_t.non <- matrix(NA, nrow = sim, ncol = p)
beta.hat_t <- matrix(NA, nrow = sim, ncol = p)
beta.hat_t.trans <- matrix(NA, nrow = sim, ncol = p)
beta.hat_t.trans.sp <- matrix(NA, nrow = sim, ncol = p)
beta.hat_t.ind <- matrix(NA, nrow = sim, ncol = p)

delta.hat_t.adp <- matrix(NA, nrow = sim, ncol = p)
beta.hat_t.adp <- matrix(NA, nrow = sim, ncol = p)
# Y.hat_t.non <- matrix(NA, nrow = sim, ncol = n_t)
# Y.hat_t <- matrix(NA, nrow = sim, ncol = n_t)
# Y.hat_t.trans <- matrix(NA, nrow = sim, ncol = n_t)
# Y.hat_t.trans.sp <- matrix(NA, nrow = sim, ncol = n_t)
# Y.hat_t.ind <- matrix(NA, nrow = sim, ncol = n_t)
# Y.hat_t.adp <- matrix(NA, nrow = sim, ncol = n_t)
T.trans <- c()
T.adp <- c()
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
  ## non-transfer
  lasso.fit_t.non <- cv.glmnet(X_t, Y_t, intercept = FALSE, nfolds = 5)
  beta.hat_t.non[s,] <- coef(lasso.fit_t.non)[-1,]
  #Y.hat_t.non[s,] <- X_t%*%beta.hat_t.non[s,]
  
  ## Oracle Trans-Lasso
  X <- rbind(X_t, X_s)
  y <- c(Y_t, Y_s)
  n.vec <- c(n_t, n_s)
  size.A0 <- 1
  beta.hat_t[s,] <- las.kA(X, y, A0 = 1:size.A0, n.vec = n.vec, l1=T)$beta.kA
  #Y.hat_t[s,] <- X_t%*%beta.hat_t[s,]
  
  ## Q-aggregation Trans-Lasso
  t0 <- Sys.time()
  prop.re1 <- Trans.lasso(X, y, n.vec, I.til = 1:5, l1 = T)
  prop.re2 <- Trans.lasso(X, y, n.vec, I.til = 6:n.vec[1], l1=T)
  beta.hat_t.trans[s,] <- (prop.re1$beta.hat + prop.re2$beta.hat) / 2
  #Y.hat_t.trans[s,] <- X_t%*%beta.hat_t.trans[s,]
  T.trans[s] <- Sys.time()-t0
  
  ## Q-aggregation with a different R.hat
  prop.sp.re1 <- Trans.lasso.sp(X, y, n.vec, I.til = 1:15, l1 = T)
  prop.sp.re2 <- Trans.lasso.sp(X, y, n.vec, I.til = 16:n.vec[1], l1=T)
  beta.hat_t.trans.sp[s,] <- (prop.sp.re1$beta.sp + prop.sp.re2$beta.sp) / 2
  #Y.hat_t.trans.sp[s,] <- X_t%*%beta.hat_t.trans.sp[s,]
  
  ## ATL
  # initial estimator
  # SCAD
  beta.tilde_t <- coef(cv.ncvreg(X_t, Y_t, intercept = FALSE, nfolds = 5, penalty=c("SCAD")))[-1]
  # ridge
  # beta.tilde_t <- coef(cv.glmnet(X_t, Y_t, intercept = FALSE, alpha = 0, nfolds = 5))[-1]
  # if (p<=n_t) {
  #   lm.fit_t <- lm(Y_t~X_t+0)
  #   beta.tilde_t <- coef(lm.fit_t)
  # } else {
  #   beta.tilde_t <- c()
  #   for (j in 1:p) {
  #     beta.tilde_t[j] <- coef(lm(Y_t~X_t[,j]+0))
  #   }
  # }
  
  fold = 3
  delta.hat_t.adp.fold <- matrix(NA, nrow = fold, ncol = p)
  beta.hat_t.ind.fold <- matrix(NA, nrow = fold, ncol = p)
  beta.hat_t.adp.fold <- matrix(NA, nrow = fold, ncol = p)
  for (k in 1:fold) {
    selector <- rep(1:fold, length.out = n_s)
    X_s.train <- X_s[selector==k,]
    X_s.test <- X_s[selector!=k,]
    Y_s.train <- Y_s[selector==k]
    Y_s.test <- Y_s[selector!=k]
    
    n_s.train <- length(Y_s.train)
    n_s.test <- length(Y_s.test)
    
    ### initial estimator
    # SCAD
    beta.tilde_s.train <- coef(cv.ncvreg(X_s.train, Y_s.train, intercept = FALSE, nfolds = 5, penalty=c("SCAD")))[-1]
    # ridge
    # beta.tilde_s.train <- coef(cv.glmnet(X_s.train, Y_s.train, intercept = FALSE, alpha = 0, nfolds = 5))[-1]
    # if (p<=n_s.train) {
    #   lm.fit_s.train <- lm(Y_s.train~X_s.train+0)
    #   beta.tilde_s.train <- coef(lm.fit_s.train)
    # } else {
    #   beta.tilde_s.train <- c()
    #   for (j in 1:p) {
    #     beta.tilde_s.train[j] <- coef(lm(Y_s.train~X_s.train[,j]+0))
    #   }
    # }
    
    # SCAD
    beta.tilde_s.test <- coef(cv.ncvreg(X_s.test, Y_s.test, intercept = FALSE, nfolds = 5, penalty=c("SCAD")))[-1]
    # ridge
    # beta.tilde_s.test <- coef(cv.glmnet(X_s.test, Y_s.test, intercept = FALSE, alpha = 0, nfolds = 5))[-1]
    # if (p<=n_s.test) {
    #   lm.fit_s.test <- lm(Y_s.test~X_s.test+0)
    #   beta.tilde_s.test <- coef(lm.fit_s.test)
    # } else {
    #   beta.tilde_s.test <- c()
    #   for (j in 1:p) {
    #     beta.tilde_s.test[j] <- coef(lm(Y_s.test~X_s.test[,j]+0))
    #   }
    # }
    
    # simultaneous estimation
    lasso.fit_s.train <- cv.glmnet(X_s.train, Y_s.train, intercept = FALSE, nfolds = 5,
                                   penalty.factor = 1/beta.tilde_s.train)
    beta.hat_s.train <- coef(lasso.fit_s.train)[-1,]
    
    # hard threshold adaptive weighting
    indicator <- c(rep(0.01,1),rep(1,p-1))
    lasso.fit_t.ind <- cv.glmnet(X_t, Y_t - X_t%*%beta.hat_s.train, intercept = FALSE, nfolds = 5,
                                 penalty.factor = indicator)
    delta.hat.ind <- coef(lasso.fit_t.ind)[-1,]
    beta.hat_t.ind.fold[k,] <- beta.hat_s.train + delta.hat.ind
    #Y.hat_t.ind[s,] <- X_t%*%beta.hat_t.ind[s,]
    
    # smooth adaptive weighting
    t1 <- Sys.time()
    weighting <- (1 / abs(beta.tilde_t-beta.tilde_s.test))^gamma
    lasso.fit_t.adp <- cv.glmnet(X_t, Y_t - X_t%*%beta.hat_s.train, intercept = FALSE, nfolds = 5,
                                 penalty.factor = weighting)
    delta.hat_t.adp.fold[k,] <- delta.hat.adp <- coef(lasso.fit_t.adp)[-1,]
    beta.hat_t.adp.fold[k,] <- beta.hat_s.train + delta.hat.adp
    #Y.hat_t.adp[s,] <- X_t%*%beta.hat_t.adp[s,]
    T.adp[s] <- Sys.time()-t1
  }
  
  beta.hat_t.ind[s,] <- colMeans(beta.hat_t.ind.fold)
  
  delta.hat_t.adp[s,] <- colMeans(delta.hat_t.adp.fold)
  beta.hat_t.adp[s,] <- colMeans(beta.hat_t.adp.fold)
}

maxdiff.hat <- c(max(colMeans(delta.hat_t.adp)[1:q]), max(colMeans(delta.hat_t.adp)[-c(1:q)]))

RMSE_nontrans <- error.hat(beta_t, beta.hat_t.non)[[1]]
RMSE_naivetrans <- error.hat(beta_t, beta.hat_t)[[1]]
RMSE_Qtrans <- error.hat(beta_t, beta.hat_t.trans)[[1]]
RMSE_Qtrans.sp <- error.hat(beta_t, beta.hat_t.trans.sp)[[1]]
RMSE_indicator <- error.hat(beta_t, beta.hat_t.ind)[[1]]
RMSE_adaptive <- error.hat(beta_t, beta.hat_t.adp)[[1]]

# PE_nontrans <- error.hat(Y_t, Y.hat_t.non)[[2]]
# PE_naivetrans <- error.hat(Y_t, Y.hat_t)[[2]]
# PE_indicator <- error.hat(Y_t, Y.hat_t.ind)[[2]]
# PE_adaptive <- error.hat(Y_t, Y.hat_t.adp)[[2]]

# summary(PE_nontrans)
# summary(PE_naivetrans)
# summary(PE_indicator)
# summary(PE_adaptive)

boxplot(RMSE_nontrans, RMSE_naivetrans, RMSE_Qtrans.sp,
        RMSE_indicator, RMSE_adaptive,
        names = c('Non-Trans',
                  'Oracle Trans-Lasso', 'Trans-Lasso',
                  'H-Ada-Trans', 'S-Ada-Trans'),
        ylab = 'RMSE')
# boxplot(PE_nontrans, PE_naivetrans, PE_indicator, PE_adaptive,
#         names = c('Non-Trans', 'Trans-Lasso', 'H-Ada-Trans', 'S-Ada-Trans'), 
#         ylab = 'Prediction Error')

summary(RMSE_nontrans)
summary(RMSE_naivetrans)
summary(RMSE_Qtrans)
summary(RMSE_Qtrans.sp)
summary(RMSE_indicator)
summary(RMSE_adaptive)

cbind(c('Non-Trans','Oracle Trans-Lasso','Trans-Lasso','Trans-Lasso.sp','H-Ada-Trans','S-Ada-Trans'),
      c(rbind(mean(RMSE_nontrans), mean(RMSE_naivetrans), mean(RMSE_Qtrans), mean(RMSE_Qtrans.sp), mean(RMSE_indicator), mean(RMSE_adaptive))))

rbind(mean(T.trans), mean(T.adp)) # computation time

cbind(c('Non-similar part','Similar part'), maxdiff.hat) # detection consistency






library(glmnet)
library(ncvreg)

ATL <- function(X_t, y_t, X_s, y_s, crossing_fitting_fold=3, threshold=c("smooth")){
  p <- ncol(X_t)
  n_s <- nrow(X_s)
  
  ### initial SCAD estimator
  beta.tilde_t <- coef(cv.ncvreg(X_t, Y_t, intercept = FALSE, nfolds = 5, penalty=c("SCAD")))[-1]

  delta.hat_t.adp.fold <- matrix(NA, nrow = crossing_fitting_fold, ncol = p)
  beta.hat_t.ind.fold <- matrix(NA, nrow = crossing_fitting_fold, ncol = p)
  beta.hat_t.adp.fold <- matrix(NA, nrow = crossing_fitting_fold, ncol = p)
  for (k in 1:crossing_fitting_fold) {
    selector <- rep(1:crossing_fitting_fold, length.out = n_s)
    X_s.train <- X_s[selector==k,]
    X_s.test <- X_s[selector!=k,]
    Y_s.train <- Y_s[selector==k]
    Y_s.test <- Y_s[selector!=k]
    
    n_s.train <- length(Y_s.train)
    n_s.test <- length(Y_s.test)
    
    ### initial SCAD estimator
    beta.tilde_s.train <- coef(cv.ncvreg(X_s.train, Y_s.train, intercept = FALSE, nfolds = 5, penalty=c("SCAD")))[-1]
    beta.tilde_s.test <- coef(cv.ncvreg(X_s.test, Y_s.test, intercept = FALSE, nfolds = 5, penalty=c("SCAD")))[-1]

    lasso.fit_s.train <- cv.glmnet(X_s.train, Y_s.train, intercept = FALSE, nfolds = 5,
                                   penalty.factor = 1/beta.tilde_s.train)
    beta.hat_s.train <- coef(lasso.fit_s.train)[-1,]
    
    if (threshold=="hard") {
      ### hard threshold adaptive weighting
      indicator <- c(rep(0.01,1),rep(1,p-1))
      lasso.fit_t.ind <- cv.glmnet(X_t, Y_t - X_t%*%beta.hat_s.train, intercept = FALSE, nfolds = 5,
                                   penalty.factor = indicator)
      delta.hat.ind <- coef(lasso.fit_t.ind)[-1,]
      beta.hat_t.ind.fold[k,] <- beta.hat_s.train + delta.hat.ind
    }
    
    if (threshold=="smooth") {
      ### smooth adaptive weighting
      gamma <- 0.05
      weighting <- (1 / abs(beta.tilde_t-beta.tilde_s.test))^gamma
      lasso.fit_t.adp <- cv.glmnet(X_t, Y_t - X_t%*%beta.hat_s.train, intercept = FALSE, nfolds = 5,
                                   penalty.factor = weighting)
      delta.hat_t.adp.fold[k,] <- delta.hat.adp <- coef(lasso.fit_t.adp)[-1,]
      beta.hat_t.adp.fold[k,] <- beta.hat_s.train + delta.hat.adp
    }
    
  }
  
  if (threshold=="hard") {
    beta.hat_t.ind <- colMeans(beta.hat_t.ind.fold)
    return(beta.hat_t.ind)
  }
  
  if (threshold=="smooth") {
    delta.hat_t.adp <- colMeans(delta.hat_t.adp.fold)
    beta.hat_t.adp <- colMeans(beta.hat_t.adp.fold)
    return(beta.hat_t.adp)
  }
}



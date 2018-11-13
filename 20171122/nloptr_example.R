library('nloptr')

## Rosenbrock Banana function and gradient in separate functions
eval_f <- function(x) {
  return( 100 * (x[2] - x[1] * x[1])^2 + (1 - x[1])^2 )
}

eval_grad_f <- function(x) {
  return( c( -400 * x[1] * (x[2] - x[1] * x[1]) - 2 * (1 - x[1]),
             200 * (x[2] - x[1] * x[1])) )
}


# initial values
x0 <- c( -1.2, 1 )

opts <- list("algorithm"="NLOPT_LD_LBFGS",
             "xtol_rel"=1.0e-8)

# solve Rosenbrock Banana function
res <- nloptr( x0=x0, 
               eval_f=eval_f, 
               eval_grad_f=eval_grad_f,
               opts=opts)
print( res )               


## Rosenbrock Banana function and gradient in one function
# this can be used to economize on calculations
eval_f_list <- function(x) {
  return( list( "objective" = 100 * (x[2] - x[1] * x[1])^2 + (1 - x[1])^2,
                "gradient"  = c( -400 * x[1] * (x[2] - x[1] * x[1]) - 2 * (1 - x[1]),
                                 200 * (x[2] - x[1] * x[1])) ) )
}

# solve Rosenbrock Banana function using an objective function that
# returns a list with the objective value and its gradient               
res <- nloptr( x0=x0, 
               eval_f=eval_f_list,
               opts=opts)
print( res )



# Example showing how to solve the problem from the NLopt tutorial.
#
# min sqrt( x2 )
# s.t. x2 >= 0
#      x2 >= ( a1*x1 + b1 )^3
#      x2 >= ( a2*x1 + b2 )^3
# where
# a1 = 2, b1 = 0, a2 = -1, b2 = 1
#
# re-formulate constraints to be of form g(x) <= 0
#      ( a1*x1 + b1 )^3 - x2 <= 0
#      ( a2*x1 + b2 )^3 - x2 <= 0

library('nloptr')


# objective function
eval_f0 <- function( x, a, b ){ 
  return( sqrt(x[2]) )
}

# constraint function
eval_g0 <- function( x, a, b ) {
  return( (a*x[1] + b)^3 - x[2] )
}

# gradient of objective function
eval_grad_f0 <- function( x, a, b ){ 
  return( c( 0, .5/sqrt(x[2]) ) )
}

# jacobian of constraint
eval_jac_g0 <- function( x, a, b ) {
  return( rbind( c( 3*a[1]*(a[1]*x[1] + b[1])^2, -1.0 ), 
                 c( 3*a[2]*(a[2]*x[1] + b[2])^2, -1.0 ) ) )
}


# functions with gradients in objective and constraint function
# this can be useful if the same calculations are needed for
# the function value and the gradient
eval_f1 <- function( x, a, b ){ 
  return( list("objective"=sqrt(x[2]), 
               "gradient"=c(0,.5/sqrt(x[2])) ) )
}

eval_g1 <- function( x, a, b ) {
  return( list( "constraints"=(a*x[1] + b)^3 - x[2],
                "jacobian"=rbind( c( 3*a[1]*(a[1]*x[1] + b[1])^2, -1.0 ), 
                                  c( 3*a[2]*(a[2]*x[1] + b[2])^2, -1.0 ) ) ) )
}


# define parameters
a <- c(2,-1)
b <- c(0, 1)

# Solve using NLOPT_LD_MMA with gradient information supplied in separate function
res0 <- nloptr( x0=c(1.234,5.678), 
                eval_f=eval_f0, 
                eval_grad_f=eval_grad_f0,
                lb = c(-Inf,0), 
                ub = c(Inf,Inf), 
                eval_g_ineq = eval_g0,
                eval_jac_g_ineq = eval_jac_g0,                
                opts = list("algorithm"="NLOPT_LD_MMA"),
                a = a, 
                b = b )
print( res0 )

# Solve using NLOPT_LN_COBYLA without gradient information
res1 <- nloptr( x0=c(1.234,5.678), 
                eval_f=eval_f0, 
                lb = c(-Inf,0), 
                ub = c(Inf,Inf), 
                eval_g_ineq = eval_g0, 
                opts = list("algorithm"="NLOPT_LN_COBYLA"),
                a = a, 
                b = b )
print( res1 )


# Solve using NLOPT_LD_MMA with gradient information in objective function
res2 <- nloptr( x0=c(1.234,5.678), 
                eval_f=eval_f1, 
                lb = c(-Inf,0), 
                ub = c(Inf,Inf), 
                eval_g_ineq = eval_g1, 
                opts = list("algorithm"="NLOPT_LD_MMA", "check_derivatives"=TRUE),
                a = a,
                b = b )
print( res2 )










##### direct()
### Minimize the Hartmann6 function
hartmann6 <- function(x) {
  n <- length(x)
  a <- c(1.0, 1.2, 3.0, 3.2)
  A <- matrix(c(10.0,  0.05, 3.0, 17.0,
                3.0, 10.0,  3.5,  8.0,
                17.0, 17.0,  1.7,  0.05,
                3.5,  0.1, 10.0, 10.0,
                1.7,  8.0, 17.0,  0.1,
                8.0, 14.0,  8.0, 14.0), nrow=4, ncol=6)
  B  <- matrix(c(.1312,.2329,.2348,.4047,
                 .1696,.4135,.1451,.8828,
                 .5569,.8307,.3522,.8732,
                 .0124,.3736,.2883,.5743,
                 .8283,.1004,.3047,.1091,
                 .5886,.9991,.6650,.0381), nrow=4, ncol=6)
  fun <- 0.0
  for (i in 1:4) {
    fun <- fun - a[i] * exp(-sum(A[i,]*(x-B[i,])^2))
  }
  return(fun)
}
S <- directL(hartmann6, rep(0,6), rep(1,6),
             nl.info = TRUE, control=list(xtol_rel=1e-8, maxeval=1000))
## Number of Iterations....: 500 
## Termination conditions:  stopval: -Inf
##     xtol_rel: 1e-08,  maxeval: 500,  ftol_rel: 0,  ftol_abs: 0 
## Number of inequality constraints:  0 
## Number of equality constraints:    0 
## Current value of objective function:  -3.32236800687327
## Current value of controls:
##     0.2016884 0.1500025 0.4768667 0.2753391 0.311648 0.6572931
 
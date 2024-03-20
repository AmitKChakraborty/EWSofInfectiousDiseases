  library(EpiEstim)
  library(ggplot2)
  
  df_EDMONTON <- read.csv("../data/empirical/COVID-19_in_Edmonton__Day_by_day.csv", header=TRUE)
  
  df_cases <- as.numeric(df_EDMONTON$Confirmed.Cases..Daily.Change)
 
  T <- nrow(df_EDMONTON)-1
  t_start <- seq(2, T-13)         #starting at 2 as conditional on the past observations
  t_end <- t_start + 13 
  
  output <- estimate_R(df_cases[2:1237],
                             method="parametric_si",
                             config = make_config(list(
                               t_start = t_start,
                               t_end = t_end,
                               mean_si = 6.3,
                               std_si = 4.2)))

  
  plot(output)
  
  output_R <- data.frame(output$R)
  output_I <- data.frame(output$I)
  merged_output <- qpcR:::cbind.na(output_I, output_R)
  
# write.csv(merged_output, file = "../data/empirical/output-COVID-ED-14.csv", row.names = FALSE) 
  

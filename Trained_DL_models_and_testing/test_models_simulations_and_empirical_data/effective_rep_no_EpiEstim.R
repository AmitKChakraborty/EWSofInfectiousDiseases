#From influenza cases of the UK, we will calculate effective reproduction number using EpiEstim package in R.
library(EpiEstim)
library(ggplot2)

#import data of influenza. change your directory of data source file
influenza.data.OWinD <- read.csv("../data/empirical/influenza-data-OWinD.csv", header=FALSE)
names(influenza.data.OWinD) <- influenza.data.OWinD[1,]

country <- "United Kingdom"

df_country <- influenza.data.OWinD[influenza.data.OWinD$Country == country, ]
df_country_cases <- df_country[, c('Date', 'All strains - All types of surveillance')] 
df_cases <- as.numeric(df_country$`All strains - All types of surveillance`)

#length of the aggregated window
dt = 7L

#mean and SD of serial interval(SI)
mean_si <- 3.6
sd_si <- 1.6

method <- "parametric_si"
config <- EpiEstim::make_config(list(mean_si = mean_si,
                                     std_si = sd_si))

output <- EpiEstim::estimate_R(incid = df_cases[1:52],
                               dt = dt,
                               dt_out = 7L,
                               recon_opt = "match",
                               iter = 10L,
                               tol = 1e-6,
                               grid = list(precision = 0.001, min = -1, max = 1),
                               method = method,
                               config = config)

plot(output)

output_R <- data.frame(output$R)
output_I <- data.frame(output$I)
merged_output <- qpcR:::cbind.na(output_I, output_R)

#export output
#write.csv(merged_output, file = "output-EpiEstim-flu-UK.csv", row.names = FALSE) 



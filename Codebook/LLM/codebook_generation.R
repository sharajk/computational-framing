#codebook format
library(readr)
library(dplyr)
library(stringr)

# Read the CSV file
frames_data <- read.csv(here::here("Codebook/LLM", "llama_frames.csv"), check.names = F, row.names = NULL)

# Function to format frames for Quarto markdown
format_frames_for_quarto <- function(data) {
  
  # Initialize output text
  output_text <- ""
  
  # Loop through each row (frame)
  for (i in 1:nrow(data)) {
    frame <- data[i, ]
    
    # Function to convert number to Roman numerals
    to_roman <- function(n) {
      roman_numerals <- c("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
                          "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX")
      if (n <= length(roman_numerals)) {
        return(roman_numerals[n])
      } else {
        return(as.character(n))  # fallback for numbers > 20
      }
    }
    
    # Create frame heading (Heading 5 in markdown)
    frame_heading <- paste0("##### Frame ", to_roman(i), ": ", gsub(" Frame","", frame$Name), "\n\n")
    
    # Create bullet points with bold keys
    description <- paste0("* **Description:** ", frame$Description, "\n")
    problem_def <- paste0("* **Problem definition:** ", frame$`Problem definition`, "\n")
    causal_attr <- paste0("* **Causal interpretations:** ", frame$`Causal attribution`, "\n")
    moral_eval <- paste0("* **Moral evaluation:** ", frame$`Moral evaluation`, "\n")
    treatment_rec <- paste0("* **Treatment recommendation:** ", frame$`Treatment recommendation`, "\n")
    example <- paste0("* **Example:** \"", frame$Example, "\" \n\n")
    
    # Combine all elements for this frame
    frame_text <- paste0(frame_heading, description, problem_def, causal_attr, 
                         moral_eval, treatment_rec, example)
    
    # Add to output
    output_text <- paste0(output_text, frame_text)
  }
  
  return(output_text)
}

# Generate the formatted text
formatted_frames <- format_frames_for_quarto(frames_data)

# Print to console (you can copy this output to your Quarto document)
cat(formatted_frames)
writeLines(formatted_frames, "Codebook/LLM/llama_frames_codebook.md")
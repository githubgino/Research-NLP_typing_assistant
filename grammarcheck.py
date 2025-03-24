from transformers import pipeline

pipe = pipeline("text2text-generation", model="tamilnlpSLIIT/varta-t5-grammar-check-error-annotated")

def do_grammar_check(text):
    # Initialize an empty string to collect corrected sentences
    corrected_text = ''
    
    # Split the input text into sentences and process each sentence
    for line in text.split('.'):
        if line.strip():  # Check if the line is not empty
            output = pipe(f"grammar: {line.strip()}")
            corrected_text += output[0]['generated_text'] + '. '
    
    # Remove the trailing space and period for the final corrected text
    corrected_text = corrected_text.strip()
    
    # Initialize an empty string to collect the final output with highlights
    final_output = ''
    
    # Split the original and corrected texts into words for comparison
    original_words = text.split(' ')
    corrected_words = corrected_text.split(' ')
    
    # Iterate through both word lists and compare
    for word1, word2 in zip(original_words, corrected_words):
        if word1 != word2:
            final_output += f'<span class="highlight">{word2}</span> '
        else:
            final_output += word1 + ' '
    
    # Return the final output with highlighted corrections
    return final_output.strip()

# Example usage
input_text = "This is an example text. It has some errors."
output_text = do_grammar_check(input_text)
print(output_text)

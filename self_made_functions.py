# File for all the self made functions
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from transformers import pipeline, AutoModel, AutoTokenizer
from prettytable import PrettyTable
from IPython.display import display
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import torch
import jiwer
import os

# Used in 'test_nb_trans.ipynb' and 'add_lost_empty_rows_to_dataframe.ipynb' --------------------------------
# Function for updating one model
def update_model(name):
    # pip install --upgrade transformers datasets huggingface_hub
    # warnings.filterwarnings(action='ignore', category=FutureWarning, message=r'.*`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`*')
    # warnings.filterwarnings(action='ignore', category=FutureWarning, message=r'.*`resume_download` is deprecated and will be removed in version 1.0.0*')
    model = AutoModel.from_pretrained(name, force_download=True) # Example for loading a model
    tokenizer = AutoTokenizer.from_pretrained(name, force_download=True) # Example for loading a tokenizer
    return model, tokenizer

# Function for updating all the models in a list
def update_all_models(model_list):
    for model in model_list:
        if model.startswith("nb"):
            model = 'NbAiLabBeta/' + model
            update_model(model)    
        else:
            model = 'openai/whisper-' + model 
            update_model(model)

# Function for checking the original datafram for a specific word or file name
def check_orig_row(df, word):
    if word.endswith('.wav'):
        row = df[df['File name'] == word]
    elif word.startswith('a') or word.startswith('d'):  
        row = df[df['File name'] == word + '.wav']
    else:
        row = df[df['Word'] == word]
    return row

# Functions to get propper whisper path
def get_whisper_path(name):
    # Handle difference between NB and standard whisper
    if name.startswith("nb"): # Put correct filepath 
        path = 'NbAiLabBeta/' + name    
    else: 
        path = 'openai/whisper-' + name
    return path
    
    
# Function to make a filename that does not exist in the directory
def get_new_csv_name(directory, base_name):
    test_for = os.path.join(directory, base_name + '.csv')
    version = 1
    csv_name = os.path.join(directory, f"{base_name}_v{version}.csv")
    # Check for existing versions and increment until a new name is found
    while os.path.isfile(csv_name):
        version += 1
        csv_name = os.path.join(directory, f"{base_name}_v{version}.csv")
    return csv_name, version

# Function for assessing what CER score correspond to which score
def percent_to_score(score):
    if score >= 0.8:
        return 1
    elif score >= 0.6:
        return 2
    elif score >= 0.4:
        return 3
    elif score >= 0.2:
        return 4
    else:
        return 5

# Function that prepare Prep the text so the calculated CER score is correct. 
# CER score calculations are case sensitive and reacts to dots at the end of the sentence
def text_prep(txt, model): 
    if not model.endswith('verbatim'):
        txt = txt.lower()
    while txt.endswith('.'):
        txt = txt[:-1]
    return txt

# Get all the different speaker ids in the dataframe
# Can be used with 'File name' and .startswith()
def get_unique_speakers(df):
    speaker_ids = []
    for i, row in df.iterrows():
        current_speaker = row['File name'][:3]
        if current_speaker not in speaker_ids:
            speaker_ids.append(current_speaker)
            
    speaker_ids = sorted(speaker_ids)
    return speaker_ids

def transcribe_one_word(path, word, model_name, out_put=False):
    if model_name.startswith("nb"):
        model_name = 'NbAiLabBeta/' + model_name
    else: 
        model_name = 'openai/whisper-' + model_name
    model = pipeline("automatic-speech-recognition", model_name)
    result = model(path, chunk_length_s=3, generate_kwargs={'task': 'transcribe', 'language': 'no'})
    
    # Evaluate transcription
    result = text_prep(result['text'], model_name)
    output = jiwer.process_characters(result, word)
    error = np.round(output.cer, 2)
    error_score = percent_to_score(output.cer)
    
    if out_put:
        print(f'Results from transcribing with {model} model')
        print(jiwer.visualize_alignment(output))
    return result, output, error, error_score

# Function for easy visual compairing of transcribed results depending on a specific file name 
def extract_row_from_csv(file_name: str, directory: str = './Transcriptions'):
    table = PrettyTable()
    table.field_names = ["Speaker ID", "Original Word", "Transcribed Word", 
                        "CER", "CER Score", "OG Word Score", "Model", "Version"] #,  "Repetition"]
    table.align['Model'] = 'l'
    
    # List all files in the directory
    dir_list = os.listdir(directory)
    idx = 0
    for csv_file in dir_list:
        # Check if the file is a CSV file and starts with 'trans'
        if csv_file.endswith('.csv') and csv_file.startswith('trans'):
            model_name = csv_file.split('_')[1]
            version = csv_file.split('_')[2].split('.c')[0]
            # print(model_name, version)
            
            csv_path = os.path.join(directory, csv_file)
            df = pd.read_csv(csv_path)
            matching_row = df[df['File name'] == file_name]
            
            speaker_id = matching_row['File name'].iloc[0][:3]
            word = matching_row['Word'].iloc[0]
            word_score = matching_row['OG Score'].iloc[0]
            cer = matching_row['CER (Character Error Rate)'].iloc[0]
            cer = np.round(cer, 2)
            cer_score = matching_row['CER Score'].iloc[0]
            transcribed_word = matching_row['Transcribed'].iloc[0]
            
            if idx == 0:
                table.add_row([speaker_id, word, transcribed_word, cer, cer_score, word_score, model_name, version])
            else: 
                table.add_row(['', '', transcribed_word, cer, cer_score, '', model_name, version])
                
            idx += 1
    return table

# Function fro transcribing one specific word for several models and ploting the results in a table
def transcribe_and_show_one_word(wn:int, df:pd.DataFrame, model_names:list, wv_path: str):# Not realy used
    table = PrettyTable()
    table.field_names = ["Model", "Speaker", "Speaker Number", "Original Word", 
                        "Transcribed Word", "CER", "CER Score", "OG Word Score", "Repetition"]

    for idx, model in enumerate(model_names):
        path = wv_path + df['File name'][wn]
        word = df['Word'][wn]
        speaker = df['File name'][wn][0]
        speaker_number = df_fin['File name'][wn][1:3]
        og_score = df['Score'][wn]
        rep = df['Repetition'][wn]
        
        result, output, error, error_score = transcribe_one_word(path=path, word=word, model_name=model, out_put=False)
        # result, output, error, error_score = transcribe_one_word(path=path, word=word, model_name=model, out_put=True)
        if model.startswith("nb"):
            model = "nb-" + model[11:]
            
        if idx == 0:
            table.add_row([model, speaker, speaker_number, word, result, error, error_score, og_score, rep])

        else:
            table.add_row([model, '', '', '', result, error, error_score, '', ''])
            
    print(table)
    return table

# Function that takes in a list of whisper model names and 
# transcribe all the words in the given dataframe.
# It saves all the results to a .csv file with version number, 
# and all empty transcriptions are also saved in correspåonding verson .csv file
def transcribe_all_audio(model_name:str, empty_path:str, data_path:str,
    # # # # # # # # # # # # # # # # # ## # # # # # # # ## # # # # # # # # # # # # 
    # --------------------------------- Problem ------------------------------- #
    # Some of the files that are not transcribed are also not in the empty file #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                    
                        df:pd.DataFrame, save: bool = False, 
                        directory:str = './Transcriptions/', highest_score_data_frame:bool = True):
    # # # # # # # # #
    # NAME HANDLING #
    # # # # # # # # #
    model_name_path = get_whisper_path(model_name)
    
    # Get the correct csv name for the transcribed file
    # From now assume we use the df with the highest score for each file name
    if highest_score_data_frame:        
        base_name = f'transcriptions_hsdf_{model_name}'
        csv_file_name, number = get_new_csv_name(directory, base_name)
        base_name_empty = os.path.join(empty_path, f'empty_hsdf_transcription_v{number}.csv')
    else:
        base_name = f'transcriptions_{model_name}'
        csv_file_name, number = get_new_csv_name(directory, base_name)
        base_name_empty = os.path.join(empty_path, f'empty_transcription_v{number}.csv')
    
    
    # print(f'File name for the transcriptions: {csv_file_name}')
    # print(f'File name for the empty transcriptions: {base_name_empty}')
    if os.path.isfile(base_name_empty):
        empty_translations_csv = pd.read_csv(base_name_empty)
        # print('Empty File already exists')
    else:
        empty_translations_csv = pd.DataFrame(columns = ['File name', 'OG word', 'idx', 'Model'])
        # print('Empty File does not exist')

    # # # # # # # # #
    #  MODEL SETUP  #
    # # # # # # # # #
    model = pipeline("automatic-speech-recognition", model_name_path) # Load the model
    # # Prosody, Noise/Disruption, Pre-speech noise, Repetition, Word, Pronunciation,
    transcribed_df = pd.DataFrame(columns=['File name', 'Word', 'Transcribed', 'CER (Character Error Rate)', 'CER Score', 'OG Score', 'CER Output'])
    new_empty_df = pd.DataFrame(columns=empty_translations_csv.columns)
    index_empty_rows = [] # Store idx to empty rows
    
    # # # # # # # # # # # #
    # MODEL TRANSCRIBING  #
    # # # # # # # # # # # #
    for i, row in df.iterrows(): # Iterate over the data frame
        wave_path = data_path + row['File name']
        transcription = model(wave_path, generate_kwargs={'task': 'transcribe', 'language': 'no'}) 
        cl_tr_txt = text_prep(transcription["text"], model_name)
        cl_tr_txt = cl_tr_txt.strip() # If whitespaces remove them
        word = row['Word'].strip()
        
        # Checks for empty strings in transcriptions 
        if transcription['text']: #---------------------------------------------------------- IF ERROR NB! CHANGED THIS FROM cl_tr_txt to transcription['text'] --------------------
            # Calculate the character error rate
            output = jiwer.process_characters(cl_tr_txt, word) # this will store the ref, hyp, alignments, CER, etc.
            error = np.round(output.cer, 2)
            # cer = jiwer.cer(cl_tr_txt, word) # only get the cer error
            error_score = percent_to_score(error)
            # Put results in the data frame
            transcribed_df.loc[i] = [row['File name'], word, cl_tr_txt, error, error_score,row['Score'], output] # Directly add a new row to the DataFrame    
        else:
            new_empty_df.loc[i] = [row['File name'], word, i, model_name] # Mer elegant å bruke annet enn i (?) 
            index_empty_rows.append(i)
            print(f'Trans String is Empty idx {i}')
            
        # if i == 2: # Used to test if the files are saved whit the correct name
        #     index_empty_rows = [1, 2, 3] 
        #     break  # Quick test to see if the code works
        
        # Have consistent prints so the server does not disconnect
        if i % 900 == 0: print(f'Index {i} of {len(df)}') 
        
    print(f'Not transcribed word for model {model_name} : ', index_empty_rows)
    print(f"Finished transcribing all the words in {model_name}\n")
    
    if save: 
        try: # save the transcriptions
            # Header: Word,Transcribed, Wer Error, Score
            transcribed_df.to_csv(csv_file_name, index=False) 

        except Exception as e: # If it can't save it for some reason print the df
            print(f'Error saving the transcribed whisper model {model_name} to csv: {e}')
            print('Transcribed df : \n', transcribed_df)
        
        version = number
        word_score = transcribed_df['OG Score']
        cer_score = transcribed_df['CER Score']
        try: #write results
            accuracy_precision_recall(word_score, cer_score, model_name, 
                version, save = True, directory = './Transcriptions/Metrics_results')
        except Exception as e:
            print(f'Error saving the metrics for the whisper model {model_name} to csv: {e}')
        try: # save the confution metrix
            conf_matrix(word_score, cer_score, model_name, version, plot = False, 
                        save = True, directory = './Transcriptions/Confution_matrix')
        except Exception as e:
            print(f'Error saving the confusion matrix for the whisper model {model_name} to csv: {e}')
        
        if index_empty_rows: # False if empty, true if not empty
            try: # save the empty strings
                print('len to new_empty_df', len(new_empty_df))
                empty_translations_csv = pd.concat([empty_translations_csv, new_empty_df], ignore_index=True)
                empty_translations_csv.to_csv(base_name_empty, index=False)
                
            except Exception as e:
                print(f'Error saving the empty translations: {e}')
                # print('New empty dataframe :\n', new_mt_df)        
        return None
    else: # Only return values if the files are not saved
        return transcribed_df, empty_translations_csv

def accuracy_precision_recall(word_score, cer_score, name, version, save = False, 
                            directory = './Transcriptions'):
    categories = [1, 2, 3, 4, 5] # Score category

    # Calculate precision, recall, and accuracy
    metrics = {'Category': [], 'Precision': [], 'Recall': [], 'Accuracy': []}

    for category in categories:
        # Makes which score row is evaluated
        y_true_score = word_score == category
        y_pred_score = cer_score == category
        # calculating the results
        precision = np.round(precision_score(y_true_score, y_pred_score), 4)
        recall = np.round(recall_score(y_true_score, y_pred_score), 4)
        accuracy = np.round(accuracy_score(y_true_score, y_pred_score), 4)
        # Storing the results
        metrics['Category'].append(category)
        metrics['Precision'].append(precision)
        metrics['Recall'].append(recall)
        metrics['Accuracy'].append(accuracy)
        
    # save the results in a .csv file
    metrics_df = pd.DataFrame(metrics)
    mean_row = pd.DataFrame({'Category': ['Mean'], 
                            'Precision': np.round(metrics_df['Precision'].mean(), 4),
                            'Recall': np.round(metrics_df['Recall'].mean(), 4), 
                            'Accuracy': np.round(metrics_df['Accuracy'].mean(), 4)}, index=[5])
    metrics_df = pd.concat([metrics_df, mean_row])
    if save:
        metrics_df.to_csv(f'{directory}/metrics_{name}_v{version}.csv', index=False)
        return None
    else:
        # Display with tabulate got from ChatGPT    
        print(tabulate(metrics_df, headers='keys', tablefmt='pretty', showindex = False))
        display(metrics_df)
        # print("\nMetrics for each category:\n", metrics_df)
        return metrics_df

def metrics_from_csv(csv_file:str): 
    metrics = pd.read_csv(csv_file)
    print(tabulate(metrics, headers='keys', tablefmt='pretty', showindex = False))
    return None

def conf_matrix(word_score, cer_score, name, version, plot = True, save = False, 
                directory = './Transcriptions'):  
    categories = [1, 2, 3, 4, 5] 
    conf_matrix = confusion_matrix(word_score, cer_score, labels=categories)
    
    if save:
        conf_matrix_df = pd.DataFrame(conf_matrix, columns=categories, index=categories)
        conf_matrix_df.to_csv(f'{directory}/conf_matrix_{name}_v{version}.csv')
    
    if plot: 
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=categories, yticklabels=categories)
        plt.xlabel('Predicted Score')
        plt.ylabel('Actual Score')
        plt.title(f'Confusion Matrix for model {name} {version}')
        plt.show()
    
    # Display confusion matrix and metrics
    print("Confusion Matrix:\n", conf_matrix)

def plot_conf_matrix_from_csv(csv_file:str):
    conf_matrix = pd.read_csv(csv_file)
    categories = [1, 2, 3, 4, 5] 
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Score')
    plt.ylabel('Actual Score')
    name = csv_file.split('_')[2]
    version = csv_file.split('_')[3].split('.c')[0]
    plt.title(f'Confusion Matrix for model {name} {version}')
    plt.show()
    return None


# Used in 'add_lost_empty_rows_to_dataframe.ipynb' --------------------------------
def find_missing_words(df1, df2): # Help from ChatGPT
    words_df1 = set(df1['File name'])  # Convert the 'words' column in to a set
    words_df2 = set(df2['File name']) 
    # Find words in df1 that are not in df2 - assume those are the empty transcriptions left out
    missing_words = words_df1.difference(words_df2)
    return missing_words # return list


# Used to get the way the data frame look like for the first part of the prodject --------------------------------
def get_df(): # This might needs to be changed based om how duplicate assesments are done
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_path = '/talebase/data/speech_raw/teflon_no/'
    wv_path = data_path + 'speech16khz/'
    
    # Data frame fixing -------------
    df_assessment = pd.read_csv(data_path+'assessments.csv') # Reed the csv assessment file 
    df_no_dup = df_assessment.drop_duplicates('File name') # Remove all the duplicate names -> nr of dups wiht this: 2821 
    df_no_zero = df_no_dup.drop(df_no_dup[df_no_dup['Score'] == 0].index) # Drop all the rows with the score 0
    df_fin = df_no_zero.reset_index(drop=True) # This resets the index after rows was dropped
    
    return df_fin, wv_path

def get_correct_df(): # Optimized version whit help from ChatGPT
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_path = '/talebase/data/speech_raw/teflon_no/'
    wv_path = data_path + 'speech16khz/'
    
    # Data frame fixing -------------
    df_new_assessment = pd.read_csv(data_path + 'assessments.csv') # Loade file
    df_new_no_zero = df_new_assessment[df_new_assessment['Score'] != 0] # Remove Zero scores
    
    # Get the index of the max score for each file name group
    max_score_indices = df_new_no_zero.groupby('File name')['Score'].idxmax() 
    df_new_no_dup = df_new_no_zero.loc[max_score_indices]
    
    # Get the mean score for each file name group
    mean_scores = df_new_no_zero.groupby('File name')['Score'].mean().apply(np.ceil).astype(int)
    
    # Replace the 'Score' column with the rounded mean scores
    df_new_no_dup['Score'] = df_new_no_dup['File name'].map(mean_scores)
    df_new_no_dup.reset_index(drop=True, inplace=True)    
    
    return df_new_no_dup, wv_path



# Calculate mean, and switch global score column

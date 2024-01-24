'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

from collections import defaultdict
import glob
import re
import scipy.io
import numpy as np
import pdb
import pandas as pd
import csv

field_names = ['No', 'Company', 'Car Model'] 

NUM_TRAINING_EXAMPLES = 4172
NUM_TEST_EXAMPLES = 1000

BASE_DIR = '../data/'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# ************* Features *************

# Features that look for certain words
def freq_pain_feature(text, freq):
    return float(freq['pain'])

def freq_private_feature(text, freq):
    return float(freq['private'])

def freq_bank_feature(text, freq):
    return float(freq['bank'])

def freq_money_feature(text, freq):
    return float(freq['money'])

def freq_drug_feature(text, freq):
    return float(freq['drug'])

def freq_spam_feature(text, freq):
    return float(freq['spam'])

def freq_prescription_feature(text, freq):
    return float(freq['prescription'])

def freq_creative_feature(text, freq):
    return float(freq['creative'])

def freq_height_feature(text, freq):
    return float(freq['height'])

def freq_featured_feature(text, freq):
    return float(freq['featured'])

def freq_differ_feature(text, freq):
    return float(freq['differ'])

def freq_width_feature(text, freq):
    return float(freq['width'])

def freq_other_feature(text, freq):
    return float(freq['other'])

def freq_energy_feature(text, freq):
    return float(freq['energy'])

def freq_business_feature(text, freq):
    return float(freq['business'])

def freq_message_feature(text, freq):
    return float(freq['message'])

def freq_volumes_feature(text, freq):
    return float(freq['volumes'])

def freq_revision_feature(text, freq):
    return float(freq['revision'])

def freq_path_feature(text, freq):
    return float(freq['path'])

def freq_meter_feature(text, freq):
    return float(freq['meter'])

def freq_memo_feature(text, freq):
    return float(freq['memo'])

def freq_planning_feature(text, freq):
    return float(freq['planning'])

def freq_pleased_feature(text, freq):
    return float(freq['pleased'])

def freq_record_feature(text, freq):
    return float(freq['record'])

def freq_out_feature(text, freq):
    return float(freq['out'])

#Features that look for certain characters
def freq_semicolon_feature(text, freq):
    return text.count(';')

def freq_dollar_feature(text, freq):
    return text.count('$')

def freq_sharp_feature(text, freq):
    return text.count('#')

def freq_exclamation_feature(text, freq):
    return text.count('!')

def freq_para_feature(text, freq):
    return text.count('(')

def freq_bracket_feature(text, freq):
    return text.count('[')

def freq_and_feature(text, freq):
    return text.count('&')

# --------- Add your own feature methods ----------
def example_feature(text, freq):
    return int('example' in text)

def freq_easy_feature(text, freq):
    return int(freq['easy'])

def freq_hundred_percent_feature(text, freq):
    return int(freq['100%'])

def freq_now_feature(text, freq):
    return int(freq['now'])

def freq_fast_feature(text, freq):
    return int(freq['fast'])

def freq_free_feature(text, freq):
    return int(freq['free'])

def freq_cash_feature(text, freq):
    return int(freq['cash'])

def freq_increase_feature(text, freq):
    return int(freq['increase'])

def freq_all_feature(text, freq):
    return int(freq['all'])

def freq_address_feature(text, freq):
    return int(freq['address'])

def freq_make_feature(text, freq):
    return int(freq['make'])

def freq_our_feature(text, freq):
    return int(freq['our'])

def freq_remove_feature(text, freq):
    return int(freq['remove'])

def freq_order_feature(text, freq):
    return int(freq['order'])

def freq_mail_feature(text, freq):
    return int(freq['mail'])

def freq_receive_feature(text, freq):
    return int(freq['receive'])

def freq_people_feature(text, freq):
    return int(freq['people'])

def freq_credit_feature(text, freq):
    return int(freq['credit'])

def freq_project_feature(text, freq):
    return int(freq['project'])

def freq_edu_feature(text, freq):
    return int(freq['edu'])

def freq_re_feature(text, freq):
    return int(freq['re'])

def freq_direct_feature(text, freq):
    return int(freq['direct'])

def freq_people_feature(text, freq):
    return int(freq['people'])

def freq_people_feature(text, freq):
    return int(freq['our'])

def freq_forward_feature(text, freq):
    return int(freq['forward'])



def freq_ect_feature(text, freq):
    return int(freq['ect'])

def freq_hou_feature(text, freq):
    return int(freq['hou'])

def freq_enron_feature(text, freq):
    return int(freq['enron'])

def freq_on_feature(text, freq):
    return int(freq['on'])

def freq_your_feature(text, freq):
    return int(freq['your'])

def freq___feature(text, freq):
    return int(freq['_'])

def freq_3_feature(text, freq):
    return int(freq['3'])

def freq_2000_feature(text, freq):
    return int(freq['2000'])

def freq_will_feature(text, freq):
    return int(freq['will'])

def freq_gas_feature(text, freq):
    return int(freq['gas'])

def freq_please_feature(text, freq):
    return int(freq['please'])

def freq_deal_feature(text, freq):
    return int(freq['deal'])

def freq_subject_feature(text, freq):
    return int(freq['subject'])

def freq_meter_feature(text, freq):
    return int(freq['meter'])

def freq_td_feature(text, freq):
    return int(freq['td'])

def freq_www_feature(text, freq):
    return int(freq['www'])

def freq_daren_feature(text, freq):
    return int(freq['daren'])

def freq_information_feature(text, freq):
    return int(freq['information'])

def freq_statement_feature(text, freq):
    return int(freq['statement'])

def freq_nbsp_feature(text, freq):
    return int(freq['nbsp'])

def freq_price_feature(text, freq):
    return int(freq['price'])

def freq_cc_feature(text, freq):
    return int(freq['cc'])

def freq_hpl_feature(text, freq):
    return int(freq['hpl'])

def freq_pm_feature(text, freq):
    return int(freq['pm'])

def freq_mmbtu_feature(text, freq):
    return int(freq['mmbtu'])

def freq_forwarded_feature(text, freq):
    return int(freq['forwarded'])

def freq_height_feature(text, freq):
    return int(freq['height'])

def freq_p_feature(text, freq):
    return int(freq['p'])

def freq_new_feature(text, freq):
    return int(freq['new'])

def freq_farmer_feature(text, freq):
    return int(freq['farmer'])

def freq_attached_feature(text, freq):
    return int(freq['attached'])

def freq_width_feature(text, freq):
    return int(freq['width'])

def freq_now_feature(text, freq):
    return int(freq['now'])

def freq_time_feature(text, freq):
    return int(freq['time'])

def freq_size_feature(text, freq):
    return int(freq['size'])

def freq_xls_feature(text, freq):
    return int(freq['xls'])

def freq_stock_feature(text, freq):
    return int(freq['stock'])

def freq_message_feature(text, freq):
    return int(freq['message'])

def freq_contract_feature(text, freq):
    return int(freq['contract'])

def freq_investment_feature(text, freq):
    return int(freq['investment'])

def freq_sitara_feature(text, freq):
    return int(freq['sitara'])

def freq_volume_feature(text, freq):
    return int(freq['volume'])

def freq_texas_feature(text, freq):
    return int(freq['texas'])

def freq_securities_feature(text, freq):
    return int(freq['securities'])

def freq_inc_feature(text, freq):
    return int(freq['inc'])

def freq_company_feature(text, freq):
    return int(freq['company'])

# Generates a feature vector
def generate_feature_vector(text, freq):
    feature = []
    feature.append(freq_pain_feature(text, freq))
    feature.append(freq_private_feature(text, freq))
    feature.append(freq_bank_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_drug_feature(text, freq))
    feature.append(freq_spam_feature(text, freq))
    # feature.append(freq_prescription_feature(text, freq))
    # feature.append(freq_creative_feature(text, freq))
    feature.append(freq_height_feature(text, freq))
    # feature.append(freq_featured_feature(text, freq))
    # feature.append(freq_differ_feature(text, freq))
    # feature.append(freq_width_feature(text, freq))
    # feature.append(freq_other_feature(text, freq))
    # feature.append(freq_energy_feature(text, freq))
    feature.append(freq_business_feature(text, freq))
    # feature.append(freq_message_feature(text, freq))
    # feature.append(freq_volumes_feature(text, freq))
    feature.append(freq_revision_feature(text, freq))
    # feature.append(freq_path_feature(text, freq))
    feature.append(freq_meter_feature(text, freq))
    feature.append(freq_memo_feature(text, freq))
    # feature.append(freq_planning_feature(text, freq))
    # feature.append(freq_pleased_feature(text, freq))
    # feature.append(freq_record_feature(text, freq))
    # feature.append(freq_out_feature(text, freq))

    # feature.append(freq_semicolon_feature(text, freq))
    feature.append(freq_dollar_feature(text, freq))
    feature.append(freq_sharp_feature(text, freq))
    feature.append(freq_exclamation_feature(text, freq))
    # feature.append(freq_para_feature(text, freq))
    # feature.append(freq_bracket_feature(text, freq))
    # feature.append(freq_and_feature(text, freq))

    # --------- Add your own features here ---------
    # Make sure type is int or float
    feature.append(freq_easy_feature(text, freq))
    # feature.append(freq_hundred_percent_feature(text, freq))
    # feature.append(freq_now_feature(text, freq))
    # feature.append(freq_fast_feature(text, freq))
    feature.append(freq_free_feature(text, freq))
    feature.append(freq_cash_feature(text, freq))
    # feature.append(freq_increase_feature(text, freq))
    # feature.append(freq_all_feature(text, freq))
    # feature.append(freq_address_feature(text, freq))
    # feature.append(freq_make_feature(text, freq))
    # feature.append(freq_our_feature(text, freq))
    # feature.append(freq_remove_feature(text, freq))
    # feature.append(freq_order_feature(text, freq))
    # feature.append(freq_mail_feature(text, freq))
    # feature.append(freq_receive_feature(text, freq))
    # feature.append(freq_credit_feature(text, freq))
    # feature.append(freq_project_feature(text, freq))
    # feature.append(freq_edu_feature(text, freq))
    # feature.append(freq_re_feature(text, freq))
    # feature.append(freq_direct_feature(text, freq))
    # feature.append(freq_people_feature(text, freq)) 
    # feature.append(freq_our_feature(text, freq))
    # feature.append(freq_forward_feature(text, freq))

    feature.append(freq_ect_feature(text, freq))
    feature.append(freq_hou_feature(text, freq))
    feature.append(freq_enron_feature(text, freq))
    feature.append(freq_on_feature(text, freq))
    feature.append(freq_your_feature(text, freq))
    feature.append(freq___feature(text, freq))
    feature.append(freq_2000_feature(text, freq))
    feature.append(freq_3_feature(text, freq))
    feature.append(freq_will_feature(text, freq))
    feature.append(freq_on_feature(text, freq))
    feature.append(freq_gas_feature(text, freq))
    feature.append(freq_please_feature(text, freq))
    feature.append(freq_deal_feature(text, freq))
    feature.append(freq_subject_feature(text, freq))
    feature.append(freq_cc_feature(text, freq))
    feature.append(freq_hpl_feature(text, freq))
    feature.append(freq_pm_feature(text, freq))
    feature.append(freq_will_feature(text, freq))
    feature.append(freq_company_feature(text, freq))
    feature.append(freq_td_feature(text, freq))
    feature.append(freq_www_feature(text, freq))
    feature.append(freq_daren_feature(text, freq))
    feature.append(freq_information_feature(text, freq))
    feature.append(freq_statement_feature(text, freq))
    feature.append(freq_nbsp_feature(text, freq))
    feature.append(freq_price_feature(text, freq))
    feature.append(freq_mmbtu_feature(text, freq))
    feature.append(freq_forwarded_feature(text, freq))
    feature.append(freq_p_feature(text, freq))
    feature.append(freq_new_feature(text, freq))
    feature.append(freq_farmer_feature(text, freq))
    feature.append(freq_attached_feature(text, freq))
    feature.append(freq_width_feature(text, freq))
    feature.append(freq_now_feature(text, freq))
    feature.append(freq_time_feature(text, freq))
    feature.append(freq_size_feature(text, freq))
    feature.append(freq_xls_feature(text, freq))
    return feature

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames):
    design_matrix = []
    word_freq = defaultdict(int)
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read() # Read in text from file
            except Exception as e:
                # skip files we have trouble reading.
                continue
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word_freq[word] += 1

            # Create a feature vector
            feature_vector = generate_feature_vector(text, word_freq)
            design_matrix.append(feature_vector)
    # print(word_freq)
    # if spam == 'spam':
    #     with open('spam.csv', 'w') as csvfile:
    #         writer = csv.DictWriter(csvfile, word_freq.keys()) 
    #         writer.writeheader() 
    #         writer.writerow(word_freq) 
    # elif spam == 'ham':
    #     with open('ham.csv', 'w') as csvfile:
    #         writer = csv.DictWriter(csvfile, word_freq.keys()) 
    #         writer.writeheader() 
    #         writer.writerow(word_freq) 
    return design_matrix

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
spam_design_matrix = generate_design_matrix(spam_filenames)
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
ham_design_matrix = generate_design_matrix(ham_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

X = spam_design_matrix + ham_design_matrix
Y = np.array([1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)).reshape((-1, 1)).squeeze()

np.savez(BASE_DIR + 'spam-data.npz', training_data=X, training_labels=Y, test_data=test_design_matrix)

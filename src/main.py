# CMPUT 501 - Assignment 3
# Ngram Language Model
# Mohammad K., Ali Z.

import sys
import re
import csv
from math import log
from collections import defaultdict
from os import listdir
from os.path import isfile, join


def log_base_2(probability):
    """ Calculates the log base two of the input probability
    Parameters:
        probability (float): The input probability of an event
    Returns:
        The log probability of the input event
    """
    if probability == 0:
        return float('-inf')
    return log(probability, 2)


def get_ngram_count(n, path):
    """ Takes the ngram count of input file based on the given n value
    Parameters:
        n (int): indicates the n in ngram for the counting
        path (string): the path to the input file to count the ngrams
    Returns:
        counts_dict (dictionary) : the dictionary count of the ngrams
        vocab (set) : the vocabulary of all unique characters

    """
    counts_dict = defaultdict(int)
    with open(path) as file:
        text_list = file.readlines()
    text_list = [re.sub('\n', ' ', line) for line in text_list]
 
    # if n is not equal to one, then for calculating the probability of some characters, we need to have
    # starting and ending symbols to enable bigram, trigram, etc. at the beginning or ending section of the sentence.
    if n != 1:
        right_pad = '<s>' * (n-1)
        left_pad = '</s>' * (n-1)
        counts_dict[right_pad] = defaultdict(int)
        counts_dict[left_pad] = defaultdict(int)
        for i in range(len(text_list)):
            counts_dict[right_pad][text_list[i][0]] += 1
            counts_dict[left_pad][text_list[i][-1]] += 1
            text_list[i] = '<s>' + text_list[i] + '</s>'

    text = ''.join(text_list)
    text = text.strip()
    uni_vocab = set(list(text))

    # if the input n is 1
    if n == 1:
        for i in range(len(text)):
            unigram = text[i]
            # if the unigram exists in the counts_dict, add one to its count
            if unigram in counts_dict.keys():
                counts_dict[unigram] += 1
            # if it's the first time seeing this, set its value to 1
            else:
                counts_dict[unigram] = 1
    # if the input n is greater than 1
    else:
        for i in range(len(text) - n + 1):
            # take up to n-1 letters of the word
            ngram_a = text[i: (i + n - 1)]
            # take the last character of the word
            ngram_b = text[(i + n - 1)]
            # if the n-1 letters of the word exists in dictionary
            if ngram_a in counts_dict.keys():
                # and the last character exists inside the nested dictionary, then add one to its count
                if ngram_b in counts_dict[ngram_a]:
                    counts_dict[ngram_a][ngram_b] += 1
                # if it was never seen before, set its value to 1
                else:
                    counts_dict[ngram_a][ngram_b] = 1
            # if it's the first time seeing this ngram, set its value to 1
            else:
                counts_dict[ngram_a] = {ngram_b: 1}

    return counts_dict, uni_vocab


def cal_perplexity_no_smoothing(n, file_path, counts):
    """ Calculates the perplexity of a sentence without smoothing, having the counts dictionary
    Parameters:
        n (int): indicates the n in ngram for the counting
        path (string): the path to the input file to count the ngrams
        counts (dictionary):the dictionary count of the ngrams
    Returns:
        perplexity (float): inverse probability (perplexity) of the given text

    """    
    probabilities = []
    text = []
    with open(file_path) as file:
        text_list = file.readlines()
    text_list = [re.sub('\n', ' ', line) for line in text_list]
    left_pad = ['<s>']*(n-1)
    right_pad = ['</s>']*(n-1)
    for i in range(len(text_list)):
        text.extend(left_pad)
        text.extend(list(text_list[i].strip()))
        text.extend(right_pad) 
    
    for i in range(len(text) - n + 1):
        # if it is about a unigram
        if n == 1:
            # take the unigram
            unigram = text[i]
            # if the unigram exists in the counts dictionary
            if unigram in counts.keys():
                # take the unigram count from the dictionary
                unigram_count = counts[unigram]
                # the the sum over all unigram counts from the dictionary
                unigram_total = sum(counts.values())
                # compute the probability of the unigram count
                probab = unigram_count / unigram_total
            # if the unigram does not exist in the counts dictionary
            else:
                probab = 0
        # for ngrams (i.e., n>1)
        else:
            ngram_a = ''.join(text[i: (i + n - 1)])
            ngram_b = ''.join(text[(i + n - 1)])
            # if the up to n-1 part of the ngram is not in the counts dictionary
            if ngram_a not in counts.keys():
                probab = 0
            # if the last character of the ngram is not in the dictionary
            elif ngram_b not in counts[ngram_a].keys():
                probab = 0
            else:
                ngram_a_count = counts[ngram_a][ngram_b]
                ngram_b_count = sum(counts[ngram_a].values())
                probab = ngram_a_count / ngram_b_count

        probabilities.append(log_base_2(probab))

    # -1 * mean of a probabilities
    # PP(W) = 2^-l where l = (1/N)(log(P(w)))
    entropy = -1*(sum(probabilities)/len(probabilities))
    perplexity = pow(2, entropy)
    return perplexity


def cal_perplexity_laplace_smoothing(n, file_path, counts, vocab):
    """ Calculates the perplexity of a sentence with Laplace smoothing, having the counts dictionary and vocabulary
    Parameters:
        n (int): indicates the n in ngram for the counting
        path (string): the path to the input file to count the ngrams
        counts (dictionary): the dictionary count of the ngrams
        vocab (set): the unique words (dictionary)
    Returns:
        perplexity (float): inverse probability (perplexity) of the given text

    """
    probabilities = []
    vocab_size = len(vocab)
    text = []
    with open(file_path) as file:
        text_list = file.readlines()
    text_list = [re.sub('\n', ' ', line) for line in text_list]
    
    left_pad = ['<s>']*(n-1)
    right_pad = ['</s>']*(n-1)
    for i in range(len(text_list)):
        text.extend(left_pad)
        text.extend(list(text_list[i].strip()))
        text.extend(right_pad) 

    for i in range(len(text) - n + 1):
        # if it is a unigram
        if n == 1:
            # take the unigram
            ngram_a = text[i]
            # if the unigram exists in the counts dictionary
            if ngram_a in counts.keys():
                # take the unigram count from the dictionary
                ngram_count = counts[ngram_a]
            # if the unigram was not in the counts dictionary
            else:
                ngram_count = 0
            ngram_total = sum(counts.values())

        # if it's an ngram
        else:
            # take up to n-1 part
            ngram_a = ''.join(text[i: (i + n - 1)])
            # take the last letter
            ngram_b = ''.join(text[(i + n-1)])
            # if the first part is not present in the dictionary
            if ngram_a not in counts.keys():
                ngram_count = 0
                ngram_total = 0
            # if the second part (last character) of the ngram is not in the dictionary
            elif ngram_b not in counts[ngram_a].keys():
                ngram_count = 0
                ngram_total = sum(counts[ngram_a].values())
            # if both of the chunks exist in the ngram
            else:
                # take the ngram count from dictionary
                ngram_count = counts[ngram_a][ngram_b]
                # take the ngram total count by summing over all ngram counts in the dictionary
                ngram_total = sum(counts[ngram_a].values())

        # add one to the ngram count
        ngram_count += 1 
        # add the vocab size to the denominator
        ngram_total += vocab_size 
        # compute the probability
        probab = ngram_count / ngram_total
        probabilities.append(log_base_2(probab))

    # calculating the perplexity is equal to: PP(W) = 2^H(W)
    # H(W) = - 1/N log2 P(W) = - 1/N log2 P(w1, w2, ..., wn)
    # Since it is for i from 1 to n, then it would be a sigma over all of the probabilities (sum)
    h = -1*(sum(probabilities)/len(probabilities))
    perplexity = pow(2, h)
    return perplexity


def cal_perplexity_interpolation_smoothing(n, lambdas, file_path, counts):
    """ Calculates the perplexity of a sentence with interpolation smoothing, having the counts dict and lambda value
    Parameters:
        n (int): indicates the n in ngram for the counting
        lambdas (float): indicates the weight (impact factor)
        path (string): the path to the input file to count the ngrams
        counts (dictionary):the dictionary count of the ngrams
    Returns:
        perplexity (float): inverse probability (perplexity) of the given text

    """
    probabilities = []
    with open(file_path) as file:
        text_list = file.readlines()
    text_list = [re.sub('\n', ' ', line) for line in text_list]
    text = ''.join(text_list)

    for i in range(len(text) - n + 1):
        probab = 0 
        for n_value in range(n): 
            partial_probability = 0
            # if it's a unigram
            if n_value + 1 == 1: 
                # take the unigram
                ngram_a = text[i] 
                if ngram_a in counts[n_value + 1].keys(): 
                    ngram_count = counts[n_value + 1][ngram_a]
                    ngram_total = sum(counts[n_value+1].values())
                    partial_probability = ngram_count / ngram_total
            # if it's an ngram
            else:
                ngram_a = ''.join(text[i: (i + n_value)])
                ngram_b = ''.join(text[(i + n_value)])

                # an unknown occurrence
                if ngram_a not in counts[n_value+1].keys():
                    partial_probability += 0
                # unknown occurrence
                elif ngram_b not in counts[n_value+1][ngram_a].keys():
                    partial_probability += 0
                # we have this ngram, let's calculate the probability
                else:
                    ngram_count = counts[n_value + 1][ngram_a][ngram_b]
                    ngram_total = sum(counts[n_value+1][ngram_a].values())
                    partial_probability = ngram_count / ngram_total

            # multiply partial probability by lambda and add to the probability list
            probab += lambdas[n_value] * partial_probability

        probabilities.append(log_base_2(probab))

    # calculating the perplexity is equal to: PP(W) = 2^H(W)
    # H(W) = - 1/N log2 P(W) = - 1/N log2 P(w1, w2, ..., wn)
    # Since it is for i from 1 to n, then it would be a sigma over all of the probabilities (sum)
    entropy = -1 * (sum(probabilities)/len(probabilities))
    perplexity = pow(2, entropy)
    return perplexity


def take_files_path(dir):
    """ Takes all file path from the input directory
     Parameters:
        path (str): input directory
    Returns:
        Path to all files of the given directory
    """
    return [dir+'/'+f for f in listdir(dir) if isfile(join(dir, f))]


def unsmoothed_language_model(train_path, dev_path, n, matches):
    """ Runs the unsmoothed language model on all of the given files
     Parameters:
         train_path (str): path to the train files
         dev_path (str): path to the development files
         n (int): n in ngram
         matches (list): a list of all matches for a given dev file with respect to all train files
    Returns: returns the best matches for the dev files
    """
    language_model = {}
    # forming the language model dict with counts of the training files
    for train_file in take_files_path(train_path):
        cnt, _ = get_ngram_count(n, train_file)
        train_file_name = train_file.split('/')[-1]
        language_model[train_file_name] = cnt

    for dev_file in take_files_path(dev_path):
        selected_file = 'There is not a match!'
        min_perplexity = float('inf')
        dev_file_name = dev_file.split('/')[-1]
        
        # Computing the perplexity with respect to the training files
        for train_file in take_files_path(train_path):
            train_file_name = train_file.split('/')[-1]
            # call to the no smoothing perplexity calculator function
            pp = cal_perplexity_no_smoothing(
                                                n, 
                                                dev_file, 
                                                language_model[train_file_name]
                                                )
            # store the file name if it has the lowest pp
            if pp < min_perplexity:
                selected_file = train_file_name
                min_perplexity = pp
        # store the best result as the match
        matches.append([selected_file, dev_file_name, min_perplexity, n])
    return matches


def laplace_smoothed_language_model(train_path, dev_path, n, matches):
    """ Runs the laplace smoothed language model on all of the given files
         Parameters:
             train_path (str): path to the train files
             dev_path (str): path to the development files
             n (int): n in ngram
             matches (list): a list of all matches for a given dev file with respect to all train files
        Returns: returns the best matches for the dev files
    """
    language_models = {}
    language_model_vocab = {}
    # forming the language model dict with counts of the training files
    for train_file in take_files_path(train_path):
        cnt, vocab = get_ngram_count(n, train_file)
        train_file_name = train_file.split('/')[-1]
        language_models[train_file_name] = cnt
        language_model_vocab[train_file_name] = vocab

    for dev_file in take_files_path(dev_path):
        selected_file = 'There is not a match!'
        min_perplexity = float('inf')
        dev_file_name = dev_file.split('/')[-1]

        # Computing the perplexity with respect to the training files
        for train_file in take_files_path(train_path):
            train_file_name = train_file.split('/')[-1]
            # call to the laplace smoothed perplexity calculator function
            pp = cal_perplexity_laplace_smoothing(
                                  n,
                                  dev_file,
                                  language_models[train_file_name],
                                  language_model_vocab[train_file_name]
                                  )

            # store the file name if it has the lowest pp
            if pp < min_perplexity:
                selected_file = train_file_name
                min_perplexity = pp
        # store the best result as the match
        matches.append([selected_file, dev_file_name, min_perplexity, n])
    return matches


def find_best_lambda(ngrams):
    """ Given the ngram counts, finds the best possible lambda using deleted interpolation
    Parameters:
    ngrams (array): flattened ngrams count

    Returns:
        computed lambdas
    """
    # initializing a lambdas list of size n    
    lambdas = [0] * len(ngrams)

    # for each ngram of maximum length
    for g in ngrams[-1]:
        probabilities = [0] * len(ngrams)

        if g == '<s>':
            continue
        elif g == '</s>':
            continue
        elif '<s>' in g:
            g = re.split('<s>', g)
            for i, item in enumerate(g):
                if item == '':
                    g[i] = '<s>'
        elif '</s>' in g:
            g = re.split('</s>', g)
            for i, item in enumerate(g):
                if item == '':
                    g[i] = '</s>'
        else:
            g = list(g)
        
        # for all of the given ngrams
        for i in range(len(ngrams)):
            if i == 0:
                last_char = g[-1]
                counts = ngrams[i]
                numerator = counts[last_char] - 1
                denominator = sum(counts.values()) - 1
                if denominator != 0:
                    probabilities[i] = numerator / denominator
                else:
                    probabilities[i] = 0
            elif i == 1:
                second_last_char = g[-2]
                last_char = g[-1]
                bigram_counts = ngrams[i]
                unigram_counts = ngrams[i-1]
                numerator = bigram_counts.get(second_last_char + last_char, 0) - 1
                denominator = unigram_counts[second_last_char] - 1
                if denominator != 0:
                    probabilities[i] = numerator / denominator
                else:
                    probabilities[i] = 0
            else:
                top_seq = ''.join(g[-(i+1):])
                bottom_seq = ''.join(g[-(i+1):-1])
                top_counts = ngrams[i]
                bottom_counts = ngrams[i-1]
                numerator = top_counts.get(top_seq, 0) - 1
                denominator = bottom_counts.get(bottom_seq, 0) - 1
                if denominator != 0:
                    probabilities[i] = numerator / denominator
                else:
                    probabilities[i] = 0

        # find the maximum probability
        max_probability = max(probabilities)
        # get the index of the max probability
        index = [i for i, j in enumerate(probabilities) if j == max_probability][0]
        # increasing the lambda by the count of ngram
        lambdas[index] += ngrams[-1][''.join(g)]

    # normalize the lambdas
    lambdas = [l / sum(lambdas) for l in lambdas]
    return lambdas


def joined_counts(counts):
    """Joins the counts of the ngram within the counts dictionary
    Parameters:
        counts (dictionary): the original dictionary used for counting

    Returns:
        grams (dictionary): the new flattened dictionary
    """
    grams = {}
    for parent_key in counts.keys():
        for child_key in counts[parent_key]:
            grams[parent_key + child_key] = counts[parent_key][child_key]
    return grams


def interploated_language_model(train_path, dev_path, n, matches):
    """ Runs the interpolated language model on all of the given files
         Parameters:
             train_path (str): path to the train files
             dev_path (str): path to the development files
             n (int): n in ngram
             matches (list): a list of all matches for a given dev file with respect to all train files
        Returns: returns the best matches for the dev files
        """
    language_models = {}
    lambdas = {}

    for train_file in take_files_path(train_path):
        train_file_name = train_file.split('/')[-1]
        language_models[train_file_name] = {}
        flattened_count = []

        # take the count for each of the ngrams
        for i in range(n):
            cnt, vocab = get_ngram_count(i + 1, train_file)
            language_models[train_file_name][i+1] = cnt

            # takes the count of the ngram
            if i + 1 == 1:
                flattened_count.append(language_models[train_file_name][i+1])
            else:
                flattened_count.append(joined_counts(language_models[train_file_name][i+1]))

        # find the lambdas for each file based on the flattened counts using deleted interpolation
        lambdas[train_file_name] = find_best_lambda(flattened_count)

    for dev_file in take_files_path(dev_path):
        selected_file = 'There is not a match!'
        min_perplexity = float('inf')
        dev_file_name = dev_file.split('/')[-1]

        # computing the perplexity with respect to the training files
        for train_file in take_files_path(train_path):
            train_file_name = train_file.split('/')[-1]

            # call to the laplace smoothed perplexity calculator function
            pp = cal_perplexity_interpolation_smoothing(
                                        n,
                                        lambdas[train_file_name],
                                        dev_file,
                                        language_models[train_file_name]
                                        )

            if pp < min_perplexity:
                selected_file = train_file_name
                min_perplexity = pp

        matches.append([selected_file, dev_file_name, min_perplexity, n])
    return matches


def find_accuracy(matches):
    """Calculates the accuracy of the LM by comparing the train and dev languages
    Parameters:
        matches (list): list of language models
    Returns:
        accuracy between language model matches
    """
    total = 0
    correct = 0
    incorrect = []
    for i in range(1, len(matches)):
        instance = matches[i]
        # check whether the training language is equal to the dev language
        if instance[0].split('.')[0] == instance[1].split('.')[0]:
            # if yes, increment the correct
            correct += 1
        else:
            incorrect += [instance]
        total += 1

    return correct / total


def find_optimal_n(train_path, dev_path, smoothing):
    """Finds the best n for each of the LMs with using a for loop and accuracy comparison
     Parameters:
         train_path (str): path to the train files
         dev_path (str): path to the development files
         smoothing (str): the smoothing mode
    Returns:
        optimal n (int): the n which resulted in highest accuracy
        max_accuracy (float): the maximum accuracy that was achieved
    """
    optimal_n = 1
    max_accuracy = 0

    if smoothing == '--bestUnsmoothed':
        for n in range(1, 20):
            matches = []
            matches = unsmoothed_language_model(train_path, dev_path, n, matches)
            accuracy = find_accuracy(matches)
            if accuracy > max_accuracy:
                optimal_n = n
                max_accuracy = accuracy
    elif smoothing == '--bestLaplace':
        for n in range(1, 20):
            matches = []
            matches = laplace_smoothed_language_model(train_path, dev_path, n, matches)
            accuracy = find_accuracy(matches)
            if accuracy > max_accuracy:
                optimal_n = n
                max_accuracy = accuracy
    elif smoothing == '--bestInterpolation':
        for n in range(1, 5):
            matches = []
            matches = interploated_language_model(train_path, dev_path, n, matches)
            accuracy = find_accuracy(matches)
            if accuracy > max_accuracy:
                optimal_n = n
                max_accuracy = accuracy
    return optimal_n, max_accuracy


def csv_output(rows, mode, output_path):
    if mode == '--unsmoothed':
        # name = 'results_dev_unsmoothed.csv'
        # for linux/macos
        if '/' in output_path:
            name = output_path.split('/')[1]
        # for windows
        else:
            name = output_path.split('\\')[1]

    elif mode == '--laplace':
        if '/' in output_path:
            name = output_path.split('/')[1]
        else:
            name = output_path.split('\\')[1]

    elif mode == '--interpolation':
        if '/' in output_path:
            name = output_path.split('/')[1]
        else:
            name = output_path.split('\\')[1]

    elif mode == '--bestUnsmoothed':
        return

    elif mode == '--bestLaplace':
        return

    elif mode == '--bestInterpolation':
        return

    else:
        print('Please select the smoothing mode carefully!')
        exit()

    # with open(output_path + '/' + name, 'w', newline='') as csv_file:
    with open(output_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(rows)


def main():
    train_path = sys.argv[1]
    dev_path = sys.argv[2]
    output_path = sys.argv[3]
    smooth_mode = sys.argv[4]

    rows = [[
            'Training_file',
            'Testing_file',
            'Perplexity',
            'N'
             ]]

    if smooth_mode == "--unsmoothed" or smooth_mode is None:
        n = 1
        matches = unsmoothed_language_model(
                            train_path,
                            dev_path,
                            n,
                            rows
                            )
        # print(matches)
    elif smooth_mode == "--laplace":
        n = 4
        matches = laplace_smoothed_language_model(
                            train_path,
                            dev_path,
                            n,
                            rows
                            )
        # print(matches)
    elif smooth_mode == '--interpolation':
        n = 3
        matches = interploated_language_model(
                                    train_path,
                                    dev_path,
                                    n,
                                    rows
                                    )
        # print(matches)
    elif smooth_mode == '--bestUnsmoothed':
        print('best N for the unsmoothed LM is: %d, with an accuracy of : %.2f' % (find_optimal_n(
                                                                        train_path,
                                                                        dev_path,
                                                                        smooth_mode
                                                                        )))
    elif smooth_mode == '--bestLaplace':
        print('best N for the laplace smoothed LM is: %d, with an accuracy of : %.2f' % (find_optimal_n(
                                                                        train_path,
                                                                        dev_path,
                                                                        smooth_mode
                                                                        )))
    elif smooth_mode == '--bestInterpolation':
        print('best N for the interpolated LM is: %d, with an accuracy of : %.2f' % (find_optimal_n(
                                                                        train_path,
                                                                        dev_path,
                                                                        smooth_mode
                                                                        )))
    else:
        print('Please select the smoothing mode carefully!')
        exit()

    # storing the results into comma separated values
    sorted_matches = sorted(rows, key=lambda x: x[1])
    csv_output(sorted_matches, smooth_mode, output_path)


if __name__ == '__main__':
    main()

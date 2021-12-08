"""
Script for Extracting Sentiment Analysis from calls.
"""

# Standards Imports
import argparse
import pickle
import shelve
import os
import json
import string
import traceback
from string import punctuation
# DL and ML Imports
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
import numpy as np
from nltk.corpus import stopwords
# FastAPI Imports
import uvicorn
from fastapi import FastAPI
from typing import List


from config import HOST, ES_INDEX, ES_DOC
from elastic import get_from_elastic, update_to_elastic_search, fetch_from_elastic_using_query
from constant import MODEL_PATH, CLASS_NAMES, LANG_CLASS_NAMES, REMOVAL_WORDS, HASH_MAP_PATH
from util import access_log, error_log, execution_time
from audio_features import Audio
from Audio_emotion import AudioEmotion

# App starts from here
app = FastAPI(debug=True)
# Convert unseen data into tokens
# with open(os.path.join(MODEL_PATH, "tokenizer.pickle"), "rb") as fp:
#     english_tokenizer = pickle.load(fp)
#
# # hindi tokenizer
# with open(os.path.join(MODEL_PATH, "hindi_tokenizer.pickle"), "rb") as fp:
#     hindi_tokenizer = pickle.load(fp)
#
# # hinglish model
# # with open(os.path.join(MODEL_PATH, "hinglish_tokenizer.pickle"), "rb") as fp:
# #     hinglish_tokenizer = pickle.load(fp)
#
# # lang tokenizer
# with open(os.path.join(MODEL_PATH, "lang_tokenizer.pickle"), "rb") as fp:
#     lang_tokenizer = pickle.load(fp)
# # Save models after Training
# english_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'weights-improvement-37-0.93.hdf5'))
# # hindi model
# hindi_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'hindi_weights-improvement-15-0.78.hdf5'))
#
# # hinglish model
# # hinglish_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'hinglish_weights-improvement-33-0.77.hdf5'))
#
# # lang model
# lang_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'lang_detect.hdf5'))
# # Recorded Prediction HashMap
# prediction_file = shelve.open(os.path.join(HASH_MAP_PATH, 'prediction'))
#
# # Stops Words Removal
# stop_words = list(stopwords.words('english'))
# stop_words.remove("you")
#
# # Ascii letters
# ALPHA_CHARACTER = string.ascii_letters
#
# PUNCTUATION = list(set(punctuation))
#
#
# def clean_unwanted_character(data, lang_code):
#     """
#     Helper function to remove unwanted non-alpha character from data
#     :param data: text
#     :return: clean_text
#     """
#     try:
#         if data:
#             if lang_code in ["en-US", "en-IN"]:
#                 data = list(data)
#                 if data[0] not in tuple(ALPHA_CHARACTER):
#                     data[0] = ""
#                     temp = ''.join(data)
#                     return clean_unwanted_character(temp, lang_code)
#
#                 if data[-1] not in tuple(ALPHA_CHARACTER):
#                     data[-1] = ""
#                     temp = ''.join(data)
#                     return clean_unwanted_character(temp, lang_code)
#                 return ''.join(data).strip()
#
#             else:
#                 data = list(data)
#                 if data[0] in tuple(PUNCTUATION):
#                     data[0] = ""
#                     temp = ''.join(data)
#                     return clean_unwanted_character(temp, lang_code)
#
#                 if data[-1] in tuple(PUNCTUATION):
#                     data[-1] = ""
#                     temp = ''.join(data)
#                     return clean_unwanted_character(temp, lang_code)
#                 return ''.join(data).strip()
#
#     except Exception as exception:
#         error_log.exception("Error occurred in clean_unwanted_character function %s \n", exception)
#
#
# def check(structure, sentiment):
#     if type(structure) is list:
#         for sentiment_key in structure:
#             for key, value in sentiment_key.items():
#                 if key == "sentiment":
#                     if value:
#                         return True
#                     else:
#                         return False
#     elif type(structure) == dict:
#         for key, value in structure.items():
#             if key == sentiment:
#                 if value:
#                     return True
#                 else:
#                     return False
#
#
# def get_ngrams(text_data, lang_code):
#     """
#      Extract n_grams from text_data.
#
#     :param
#         text_data : Input text in form of str.
#         lang_code: hindi or english
#     :return:
#         Dict : N-gram dictionary consists of 1 & 2 gram
#     """
#     try:
#         data_list = []
#         # Ngram Preprocessing
#         if lang_code in ["en-US", "en-IN"]:
#             sentence_split = text_data.split('.')
#         else:
#             sentence_split = text_data.split('।')
#         sentence_split = ' '.join(sentence_split)
#         sentence_split = sentence_split.split(' ')
#         clean_one_gram = list(filter(lambda unwanted_words: unwanted_words != '', sentence_split))
#         data_list.extend(clean_one_gram)
#         data_list.extend(' '.join(idx) for item in [' '.join(clean_one_gram)]  # 2 gram extraction
#                          for idx in zip(item.split(" ")[:-1], item.split(" ")[1:]))
#         return data_list
#     except Exception as exception:
#         error_log.exception("Error occurred in get_grams function %s \n", exception)
#
#
# def lang_prediction(data):
#     """
#     Detect english and Hinglish language
#     :param data: text
#     :return: en/hi
#     """
#     token = lang_tokenizer.texts_to_sequences([data])
#     token_with_padding = sequence.pad_sequences(token, maxlen=160, dtype='int32')
#     prediction = lang_model.predict(token_with_padding, batch_size=1)
#     sentiment_class = int(np.argmax(prediction))
#     language = LANG_CLASS_NAMES[sentiment_class]
#     return language
#
#
# def model_prediction(data, lang_code):
#     """
#
#     :param data: Sentence or ngram as a input
#     :param lang_code: hindi or english
#     :return: Probable class Positive, Negative, Neutral
#     """
#     # ["en-US", "en-IN"]
#     try:
#         if data:
#             if lang_code == "en-US":
#                 # print("Enter in english model en-US")
#                 token = english_tokenizer.texts_to_sequences([data])
#                 token_with_padding = sequence.pad_sequences(token, maxlen=160, dtype='int32')
#                 prediction = english_model.predict(token_with_padding, batch_size=1)
#             elif lang_code == "en-IN":
#                 language = lang_prediction(data)
#                 if language == "en":
#                     token = english_tokenizer.texts_to_sequences([data])
#                     token_with_padding = sequence.pad_sequences(token, maxlen=160, dtype='int32')
#                     prediction = english_model.predict(token_with_padding, batch_size=1)
#                 else:
#                     return None
#                     # print("Enter in higlish model en-IN ")
#                     # token = hinglish_tokenizer.texts_to_sequences([data])
#                     # token_with_padding = sequence.pad_sequences(token, maxlen=160, dtype='int32')
#                     # prediction = hinglish_model.predict(token_with_padding, batch_size=1)
#                     # sentiment_class = int(np.argmax(prediction))
#                     # if int(max(prediction[0] * 100)) > 70:
#                     #     return {"sentiment": CLASS_NAMES[sentiment_class],
#                     #             "percentage": "{:.2f}".format(max(prediction[0] * 100))}
#             else:
#                 # print("Enter in hindi model hi-IN ")
#                 token = hindi_tokenizer.texts_to_sequences([data])
#                 token_with_padding = sequence.pad_sequences(token, maxlen=160, dtype='int32')
#                 prediction = hindi_model.predict(token_with_padding, batch_size=1)
#
#             sentiment_class = int(np.argmax(prediction))
#             if int(max(prediction[0] * 100)) > 90:
#                 return {"sentiment": CLASS_NAMES[sentiment_class],
#                         "percentage": "{:.2f}".format(max(prediction[0] * 100))}
#             return None
#
#     except Exception as exception:
#         error_log.exception("Error occurred in model_prediction function %s \n", exception)
#
#
# def n_gram_generation_check(sentence, sentiment_result, lang_code):
#     """
#     Extract ngram only if sentiment is negative and positive
#     :param sentence: Transcript
#     :param sentiment_result: sentiment result from model
#     :param lang_code: hindi or english
#     :return: N gram
#     """
#     try:
#
#         n_gram = get_ngrams(sentence, lang_code)
#         sentiment_result["positive"] = []
#         sentiment_result["negative"] = []
#         for gram in n_gram:
#             gram_split = gram.split(' ')
#             if (len(gram_split) == 1 and gram_split[0].lower() not in stop_words) or \
#                     (len(gram_split) > 1 and all(val.lower() not in stop_words for val in gram_split)):
#                 if prediction_file.get(gram.lower(), None):
#
#                     gram_lookup = prediction_file.get(gram.lower()).split('_')  # Eg 98.02_Neutral
#                     if gram_lookup[1]:
#                         sentiment_result[gram_lookup[1]].append(gram)
#                 else:
#                     gram = clean_unwanted_character(gram, lang_code)
#                     gram_prediction = model_prediction(gram, lang_code)
#                     if gram_prediction and gram_prediction.get('sentiment') \
#                             in ("positive", "negative"):
#                         sentiment_result[gram_prediction.get('sentiment')].append(gram)
#                         combined_result = gram_prediction.get("percentage") \
#                             + '_' + gram_prediction.get('sentiment')
#                         prediction_file[gram.lower()] = combined_result
#         return sentiment_result
#     except Exception as exception:
#         error_log.exception("Error occurred in n_gram_generation_check function %s \n", exception)
#
#
# def sentiment_prediction(full_sentence, lang_code):
#     """
#
#     Extract sentiment at N-gram and sentence level
#     :param full_sentence: Contains N-grams generated from get_ngrams() function
#     :param lang_code: hindi or english
#     :return: List of dict with N-gram and sentence sentiment
#     """
#     try:
#         sentiment_result = {}
#         final_result_list = []
#         if lang_code in ["en-US", "en-IN"]:
#             result_data = full_sentence.split('.')
#         else:
#             result_data = full_sentence.split('।')
#         result_data = [val.strip() for val in result_data]
#         full_transcript_sentiment = []
#         for sentence in result_data:
#             sentiment_result["positive"] = []
#             sentiment_result["negative"] = []
#             if sentence.lower() != full_sentence.replace(".", "").lower().strip():
#                 internal_sentence_prediction_from_model = model_prediction(sentence.strip(), lang_code)
#                 if internal_sentence_prediction_from_model and internal_sentence_prediction_from_model \
#                         .get('sentiment') in ["positive", "negative"]:
#                     sentiment_result["text"] = sentence
#                     sentiment_result["sentiment"] = internal_sentence_prediction_from_model.get("sentiment")
#                     sentiment_result["percentage"] = internal_sentence_prediction_from_model.get("percentage")
#                     if lang_code in ["en-US", "hi-IN"]:
#                         n_gram_generation_check(sentence, sentiment_result, lang_code)
#                     elif lang_code == "en-IN":
#                         lang = lang_prediction(sentence)
#                         if lang == "en":
#                             n_gram_generation_check(sentence, sentiment_result, lang_code)
#             # Removing empty keys
#             sentiment_result = {k: v for k, v in sentiment_result.items() if v}
#             if sentiment_result:
#                 final_result_list.append(sentiment_result)
#                 sentiment_result = {}
#
#         if prediction_file.get(full_sentence):
#             full_sentence_lookup = prediction_file.get(full_sentence.lower()).split('_')
#             if full_sentence_lookup[1]:
#                 final_result_list.append({"full_sentence": full_sentence,
#                                           "full_sentence_sentiment": full_sentence_lookup[1],
#                                           "full_sentence_percentage": full_sentence_lookup[0]})
#                 full_transcript_sentiment.append(full_sentence_lookup[1])
#
#         else:
#             for word in REMOVAL_WORDS:
#                 if word.lower() not in full_sentence.lower():
#                     full_sentence_prediction_from_model = model_prediction(full_sentence.strip(), lang_code)
#                     if full_sentence_prediction_from_model:
#                         full_transcript_sentiment.append(full_sentence_prediction_from_model.get('sentiment'))
#                     if full_sentence_prediction_from_model and full_sentence_prediction_from_model. \
#                             get('sentiment') in ["positive", "negative"]:
#                         if not final_result_list:
#                             sentiment_result["positive"] = []
#                             sentiment_result["negative"] = []
#                             if lang_code in ["en-US", "hi-IN"]:
#                                 n_gram_generation_check(full_sentence, sentiment_result, lang_code)
#                             elif lang_code == "en-IN":
#                                 lang = lang_prediction(full_sentence)
#                                 if lang == "en":
#                                     n_gram_generation_check(full_sentence, sentiment_result, lang_code)
#                         if (check(final_result_list, full_sentence_prediction_from_model.get('sentiment'))) or \
#                                 (check(sentiment_result, full_sentence_prediction_from_model.get('sentiment'))):
#                             final_result_list.append(
#                                 {"full_sentence": full_sentence,
#                                  "full_sentence_sentiment": full_sentence_prediction_from_model.get("sentiment"),
#                                  "full_sentence_percentage": full_sentence_prediction_from_model.get("percentage")})
#                         # Removing empty keys
#                         sentiment_result = {k: v for k, v in sentiment_result.items() if v}
#                         if sentiment_result:
#                             final_result_list.append(sentiment_result)
#         return final_result_list
#     except Exception as exception:
#         error_log.exception("Error occurred in sentiment_prediction function %s \n", exception)
#
#
# def sentiment_parser(data: list) -> dict:
#     """
#     Parsing sentiment prediction to positive and negative keywords
#     """
#     try:
#         response = {}
#         response["positive"] = []
#         response["negative"] = []
#         for multiple_dict in data:
#             for key, value in multiple_dict.items():
#                 if key in ["positive", "negative"]:
#                     response[key].extend(value)
#         return response
#     except Exception as exception:
#         error_log.exception("Error occurred in sentiment_parser function %s \n", exception)
#
#
# def segregation_dict(channel_type, segregation_data_dict, sentiment_result):
#     """
#     Segregate dict on the basis of agent and customer.
#     """
#     try:
#         if channel_type:
#             if sentiment_result:
#                 response = sentiment_parser(sentiment_result)
#                 if response:
#                     if segregation_data_dict.get(channel_type):
#                         if response.get("negative"):
#                             if segregation_data_dict.get(channel_type).get('negative'):
#                                 segregation_data_dict[channel_type]['negative'].extend(response.get("negative"))
#                             else:
#                                 segregation_data_dict[channel_type]['negative'] = response.get("negative")
#                         if response.get("positive"):
#                             if segregation_data_dict.get(channel_type).get('positive'):
#                                 segregation_data_dict[channel_type]['positive'].extend(response.get("positive"))
#                             else:
#                                 segregation_data_dict[channel_type]['positive'] = response.get("positive")
#                     else:
#                         segregation_data_dict[channel_type] = response
#                 return segregation_data_dict
#     except Exception as exception:
#         error_log.exception("Error occurred in segregation_dict function %s \n", exception)
#
#
# def full_transcript_analysis(sentiment_result):
#     """
#     Extract full sentiment analysis
#     :param sentiment_result: sentiment result
#     :return: neutral, positive, negative sentiment
#     """
#     try:
#         if sentiment_result:
#             sentiment_data = list(filter(lambda sentiment: sentiment.get('full_sentence_sentiment') in
#                                          ['positive', 'negative'], sentiment_result))
#             if sentiment_data:
#                 sentiment = sentiment_data[0].get("full_sentence_sentiment")
#                 if sentiment in ["positive", "negative"]:
#                     return sentiment
#             return "neutral"
#         return "neutral"
#     except Exception as exception:
#         error_log.exception("Error occurred in full_transcript_analysis function %s \n", exception)
#
#
# @execution_time
# def preprocess_and_sentiment(data: dict, call_id) -> dict:
#     """
#     Preprocess sentiment data and extract sentiment data
#     :param data:
#     :param call_id:
#     :return:sentiment_data, full_transcript_analysis and segregation_dict
#     """
#     try:
#         segregation_data_dict = {}
#         final_response = {}
#         full_transcript_sentiment = {}
#         neutral, negative, positive = 0, 0, 0
#         access_log.info("Started Preprocessing for sentiment data and sentiment analysis %s.\n", call_id)
#         result = data.get('_source', {}).get('preprocess_data', {}).get('results', [])
#         lang_code = data.get('_source', {}).get('config', {}).get('language_code')
#         for alternative_dict in result:
#             alternatives = alternative_dict.get('alternatives', [{}])[0]
#             transcript = alternatives.get('transcript', {})
#             sentiment_result = sentiment_prediction(transcript, lang_code)
#             alternatives.update({'sentiment': sentiment_result})
#             analysis = full_transcript_analysis(sentiment_result)
#             if analysis == "neutral":
#                 neutral += 1
#             elif analysis == "negative":
#                 negative += 1
#             else:
#                 positive += 1
#
#             channel_type = alternative_dict.get('channelType', {})
#             if channel_type:
#                 segregation_dict(channel_type, segregation_data_dict, sentiment_result)
#         access_log.info("Completed extraction of sentiment_segregation list for negative and positive words")
#         final_response['preprocess_data'] = {'results': result}
#         final_response['sentiment_segregation'] = segregation_data_dict
#         total_sentence = neutral + negative + positive
#         full_transcript_sentiment["neutral"] = round((neutral / total_sentence) * 100, 2) if neutral > 0 else 0
#         full_transcript_sentiment["negative"] = round((negative / total_sentence) * 100, 2) if negative > 0 else 0
#         full_transcript_sentiment["positive"] = round((positive / total_sentence) * 100, 2) if positive > 0 else 0
#         final_response['full_transcript_sentiment'] = full_transcript_sentiment
#         return final_response
#     except Exception as exception:
#         error_log.exception("Error occurred in preprocess_and_sentiment function %s \n", exception)


def generate_call_id(user_id):
    """
    Extract call_id from "fetch_from_elastic_using_query" es query
    :param user_id: id of user
    :return:
    """
    query = {"size": 1000,
             "_source": {"includes": "preprocess_data.results.alternatives.transcript"},
             "query": {
                 "bool": {
                     "must": [
                         {
                             "match_phrase": {
                                 "user_id.keyword": user_id
                             }
                         },
                     ]
                 }
             }
             }

    call_list = []
    try:
        response = fetch_from_elastic_using_query(es_index=ES_INDEX, es_doc_type=ES_DOC, query=query)
        if response.get("status_code") == 200:
            with open(os.path.join(os.getcwd(), f"call_data_{user_id}.json"), "w") as file:
                file.write(json.dumps(response, indent=4))
        else:
            print("Some Error while fetching from ES....")
        with open(os.path.join(os.getcwd(), f"call_data_{user_id}.json"), "r") as file_data:
            json_data = json.load(file_data)

        result = json_data.get('hits', {}).get("hits", {})
        for id_extraction in range(len(result)):
            for call_id, call_id_value in result[id_extraction].items():
                if call_id == "_id":
                    call_list.append(call_id_value)
        os.remove(os.path.join(os.getcwd(), f"call_data_{user_id}.json"))
        return call_list

    except Exception as exception:
        error_log.exception("Error occurred in generate_call_id function %s \n", exception)
        return None


# @app.get("/account/{user_id}")
# def sentiment_extraction_on_account(user_id: str):
#     """
#     Main API extract sentiment for entire account
#     :param user_id:
#         User_id
#     :return:
#        List of Success/Fail of all calls.
#     """
#     access_log.info("======================================\n")
#     access_log.info("Sentiment Analysis started on account %s \n", user_id)
#     all_call_ids = generate_call_id(user_id)
#     if all_call_ids:
#         success = []
#         fail = []
#         try:
#             for index, call_id in enumerate(all_call_ids):
#                 access_log.info("==================================\n")
#                 access_log.info("%d Sentiment Analysis started %s\n", index, call_id)
#                 data = get_from_elastic(ES_INDEX, ES_DOC, call_id)
#                 if data.get('status_code') == 200:
#                     updated_result = preprocess_and_sentiment(data, call_id)
#                     elastic_update = update_to_elastic_search(ES_INDEX, ES_DOC, updated_result, call_id)
#                     if elastic_update.get('status_code') == 200:
#                         success.append(("success", call_id))
#                         access_log.info("%d Extraction completed for %s successfully.\n", index, call_id)
#                         access_log.info("===============================================================\n")
#                     else:
#                         fail.append(("failure", call_id))
#                         access_log.info("%d Extraction failed for %s .\n", index, call_id)
#                         access_log.info("===============================================================\n")
#             access_log.info("===============================================================\n")
#             access_log.info("Sentiment Analysis completed on account %s \n", user_id)
#             return {"response": [success, fail]}
#         except Exception as exception:
#             error_log.exception("Error occurred in sentiment_extraction_on_account function %s \n", exception)
#             return {"response": "failure"}
#     return {"status": "failure"}
#

# @app.get("/sentiment/{call_id}")
# def sentiment_extraction(call_id: str):
#     """
#     Main API call on the basis of call_id
#     :param call_id:
#         ID of call
#     :return:
#         Success/Fail of call
#     """
#     access_log.info("========================================\n")
#     access_log.info("Sentiment Analysis started %s.\n", call_id)
#     try:
#
#         data = get_from_elastic(ES_INDEX, ES_DOC, call_id)
#         if data.get('status_code') == 200:
#             updated_result = preprocess_and_sentiment(data, call_id)
#             elastic_update = update_to_elastic_search(ES_INDEX, ES_DOC, updated_result, call_id)
#             if elastic_update.get('status_code') == 200:
#                 access_log.info("Extraction completed for %s successfully.\n", call_id)
#                 access_log.info("==============================================\n")
#                 return {"status": "success", "call_id": call_id}
#             else:
#                 access_log.info("Extraction failed for %s .\n", call_id)
#                 access_log.info("==============================================\n")
#                 return {"status": "failure", "call_id": call_id}
#         return {"status": "failure", "call_id": call_id}
#     except Exception as exception:
#         error_log.exception("Error occurred in sentiment_extraction function %s \n", exception)
#         return {"status": "failure", "call_id": call_id}
#

@app.get("/audio_features/{conversation_id}")
def get_audio_features(conversation_id: str):
    """
    Main function to evaluate audio features
    viz. Pitch, Amplitude, Frequency, Tempo etc.
    In
    """
    access_log.info(f"Audio Features Analysis started for conversation_id: {conversation_id}")
    resp = {"status": "failure", "message": "some error occurred", "statusCode": 500, "audioFeatures": {}}
    try:
        audio = Audio(conversation_id)
        audio.eval_audio_features()
        audio_features_dict = audio.get_features_dict()
        audio.remove()

        # Update in DB
        query = {'audioFeature': audio_features_dict}
        es_response = update_to_elastic_search(ES_INDEX, ES_DOC, query, conversation_id)
        if es_response['es_status'] == 'success':
            resp['status'] = 'success'
            resp['message'] = 'Audio features evaluated successfully.'
            resp['statusCode'] = 201
            resp['audioFeatures'] = audio_features_dict
        else:
            resp['message'] = 'Error occurred while updating data in Elasticsearch.'
    except:
        error_log.exception(f"Error occurred for conversation_id: {conversation_id}")
        error_log.exception(traceback.print_exc())
    return resp


@app.get("/account/audio_features/{user_id}")
def get_audio_features_for_account(user_id: str):
    """
    Main function to evaluate audio features
    viz. Pitch, Amplitude, Frequency, Tempo etc.
    for entire account
    """
    access_log.info(f"Audio Features Analysis started for Account user_id: {user_id}")
    resp = {"status": "failure", "message": "some error occurred", "statusCode": 500,
            "audioFeaturesStatus": {'success': {'count': 0, 'ids': []}, 'failure': {'count': 0, 'ids': []}}}
    try:
        all_call_ids = generate_call_id(user_id)
        if all_call_ids:
            total = len(all_call_ids)
            for index, conversation_id in enumerate(all_call_ids, start=1):
                access_log.info(f"({index}/{total}) Audio Features started for conversation_id: {conversation_id}")
                try:
                    # get audio features
                    audio = Audio(conversation_id)
                    audio.eval_audio_features()
                    audio_features_dict = audio.get_features_dict()
                    audio.remove()
                    # Update in DB
                    query = {'audioFeature': audio_features_dict}
                    es_response = update_to_elastic_search(ES_INDEX, ES_DOC, query, conversation_id)
                    if es_response['es_status'] == 'success':
                        access_log.info(f"Evaluation status for conversation_id {conversation_id}: SUCCESS")
                        resp['audioFeaturesStatus']['success']['count'] += 1
                        resp['audioFeaturesStatus']['success']['ids'].append(conversation_id)
                    else:
                        resp['message'] = 'Error occurred while updating data in Elasticsearch.'
                        access_log.info(f"Evaluation status for conversation_id {conversation_id}: FAILURE")
                        resp['audioFeaturesStatus']['failure']['count'] += 1
                        resp['audioFeaturesStatus']['failure']['ids'].append(conversation_id)
                except:
                    resp['audioFeaturesStatus']['failure']['count'] += 1
                    resp['audioFeaturesStatus']['failure']['ids'].append(conversation_id)
                    error_log.exception(f"Error occurred for conversation_id: {conversation_id}")
                    error_log.exception(traceback.print_exc())
        resp['status'] = 'success'
        resp['statusCode'] = 200
        resp['message'] = 'Audio Features updated for the conversations for the given account.'
    except:
        error_log.exception(traceback.print_exc())
    return resp



@app.get("/audio_emotion/{conversation_id}")
def get_audio_emotion(conversation_id: str):
    """
    Main function to evaluate audio emotion
    viz. Angry, Unhappy, Happy, Neutral
    """
    access_log.info(f"Audio Emotion Analysis started for conversation_id: {conversation_id}")
    resp = {"status": "failure", "message": "some error occurred", "statusCode": 500, "audioEmotion": {}}
    try:
        audio = AudioEmotion(conversation_id)
        audio_emotion = audio.eval_audio_emotion()
        audio.remove()
        print(audio_emotion)
        # Update in DB
        # query = {'audioEmotion': audio_emotion}
        # es_response = update_to_elastic_search(ES_INDEX, ES_DOC, query, conversation_id)
    #     if es_response['es_status'] == 'success':
    #         resp['status'] = 'success'
    #         resp['message'] = 'Audio Emotion evaluated successfully.'
    #         resp['statusCode'] = 201
    #         resp['audioEmotion'] = audio_emotion
    #     else:
    #         resp['message'] = 'Error occurred while updating data in Elasticsearch.'
    except:
        error_log.exception(f"Error occurred for conversation_id: {conversation_id}")
        error_log.exception(traceback.print_exc())
    return resp

@app.get("/account/audio_features/{user_id}")
def get_audio_Emotion_for_account(user_id: str):
    """
    Main function to evaluate audio emotion
    viz. Angry, Unhappy, Happy, Neutral
    for entire account
    """
    access_log.info(f"Audio Emotion Analysis started for Account user_id: {user_id}")
    resp = {"status": "failure", "message": "some error occurred", "statusCode": 500,
            "audioEmotionStatus": {'success': {'count': 0, 'ids': []}, 'failure': {'count': 0, 'ids': []}}}
    try:
        all_call_ids = generate_call_id(user_id)
        if all_call_ids:
            total = len(all_call_ids)
            for index, conversation_id in enumerate(all_call_ids, start=1):
                access_log.info(f"({index}/{total}) Audio Emotion started for conversation_id: {conversation_id}")
                try:
                    # get audio features
                    audio = AudioEmotion(conversation_id)
                    audio_emotion = audio.eval_audio_emotion()
                    audio.remove()
                    # Update in DB
                    query = {'audioEmotion': audio_emotion}
                    es_response = update_to_elastic_search(ES_INDEX, ES_DOC, query, conversation_id)
                    if es_response['es_status'] == 'success':
                        access_log.info(f"Evaluation status for conversation_id {conversation_id}: SUCCESS")
                        resp['audioEmotionStatus']['success']['count'] += 1
                        resp['audioEmotionStatus']['success']['ids'].append(conversation_id)
                    else:
                        resp['message'] = 'Error occurred while updating data in Elasticsearch.'
                        access_log.info(f"Evaluation status for conversation_id {conversation_id}: FAILURE")
                        resp['audioEmotionStatus']['failure']['count'] += 1
                        resp['audioEmotionStatus']['failure']['ids'].append(conversation_id)
                except:
                    resp['audioEmotionStatus']['failure']['count'] += 1
                    resp['audioEmotionStatus']['failure']['ids'].append(conversation_id)
                    error_log.exception(f"Error occurred for conversation_id: {conversation_id}")
                    error_log.exception(traceback.print_exc())
        resp['status'] = 'success'
        resp['statusCode'] = 200
        resp['message'] = 'Audio Emotion updated for the conversations for the given account.'
    except:
        error_log.exception(traceback.print_exc())
    return resp







if __name__ == '__main__':
    # uvicorn.run(app, host='127.0.0.1', port=8000)
    parser = argparse.ArgumentParser(description="To Run Server")
    parser.add_argument('--port', required=True)
    args = parser.parse_args()
    uvicorn.run(app, host=HOST, port=int(args.port))

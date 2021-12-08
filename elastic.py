import requests
import json

ES_ENDPOINT = "https://search-asrtesing-pvdc5l7vijkrdlnkcepb6vtrja.ap-south-1.es.amazonaws.com:443"


def get_from_elastic(es_index, es_doc_type, id):
    index = es_index
    doc = es_doc_type
    url = "{}/{}/{}/{}".format(ES_ENDPOINT, index, doc, id)
    payload = {}
    json_data = {}
    headers = {
        'Content-Type': 'application/json'
    }
    try:
        response = requests.request("GET", url, headers=headers, data=json.dumps(payload))
        status_code = response.status_code
        if status_code >= 200 and status_code <= 299:
            data = response.text.encode('utf8')
            json_data = json.loads(data)
            json_data.update({"es_status": "success", "status_code": status_code})
        else:
            json_data = {"es_status": "failure", "status_code": status_code,
                         "message": "Unable to Fetch Data from Elastic"}

    except Exception as e:
        print("Exception Occured at get_from_elastic() : ", e)
        json_data.update(
            {"es_status": "failure", "message": "Exception Occured !! Unable to Fetch Data from Elastic",
             "exception": str(e)})

    return json_data


def update_to_elastic_search(es_index, es_doc_type, query, id):

    index = es_index
    doc = es_doc_type
    url = "{}/{}/{}/{}/_update".format(ES_ENDPOINT, index, doc, id)
    json_data = {}
    payload = {
        "doc": query
    }
    headers = {
        'Content-Type': 'application/json'
    }
    try:
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
        status_code = response.status_code

        if status_code >= 200 and status_code <= 299:
            data = response.text.encode('utf8')
            json_data = json.loads(data)
            json_data.update({"es_status": "success", "status_code": status_code})
        else:
            data = response.text.encode('utf8')
            json_data = json.loads(data)
            print(json_data)
            json_data = {"es_status": "failure", "status_code": status_code,
                         "message": "Unable to Update Data to Elastic"}
    except Exception as e:
        print("Exception in update_to_elastic_search : {}".format(str(e)))
        json_data.update(
            {"es_status": "failure", "message": "Exception Occured !! Unable to Update Data to Elastic",
             "exception": str(e)})

    return json_data


def fetch_from_elastic_using_query(es_index, es_doc_type, query):
    index = es_index
    doc = es_doc_type
    url = "{}/{}/{}/_search".format(ES_ENDPOINT, index, doc)

    payload = query
    json_data = {}
    headers = {
        'Content-Type': 'application/json'
    }
    try:
        response = requests.request("GET", url, headers=headers, data=json.dumps(payload))
        status_code = response.status_code

        if status_code >= 200 and status_code <= 299:
            data = response.text.encode('utf8')
            json_data = json.loads(data)
            json_data.update({"es_status": "success", "status_code": status_code})
        else:
            json_data = {"es_status": "failure", "status_code": status_code,
                         "message": "Unable to Fetch Data from Elastic"}

    except Exception as e:
        print("Exception Occured at fetch_from_elastic_using_query() : ", e)
        json_data.update({"es_status": "failure", "message": "Exception Occured !! Unable to Fetch Data from Elastic",
                          "exception": str(e)})

    return json_data

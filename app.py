from flask import Flask, jsonify, request;
from handler import *;
import pandas as pd
from time import perf_counter;
import json

app = Flask(__name__)

data = pd.read_csv('./data/ECG_SubsequenceWTimeStamp.csv', nrows=1000)

@app.route('/similar-search', methods=['POST'])
def get_similar_search():
    parsedSequence = json.loads(request.data.decode('ASCII')).get('data').split(',')
    parsedSequence.pop(0)
    parsedSequence[len(parsedSequence)-1] = parsedSequence[len(parsedSequence)-1][:-1]
    
    for i in range(len(parsedSequence)):
        parsedSequence[i] = float(parsedSequence[i])

    resp = parallelComputing(parsedSequence, data, 5)
    print(resp)
    return jsonify(resp)

@app.route('/')
def hello_world():
    return "<h1>Similar Search Algo</h1>"

if __name__ == '__main__':
    # print(get_vernacular_list(0))
    # app.run(host='127.0.0.2', port=5000)
    app.run(debug=True)

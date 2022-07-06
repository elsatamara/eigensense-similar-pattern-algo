from flask import Flask, jsonify;
from handler import *;
import pandas as pd
from time import perf_counter;

app = Flask(__name__)

data = pd.read_csv('./data/ECG_SubsequenceWTimeStamp.csv', nrows=1000)

@app.route('/similar-search/<sequence>')
def get_similar_search(sequence):
    parsedSequence = str(sequence).split(",")
    parsedSequence.pop(0)
    parsedSequence[len(parsedSequence)-1] = parsedSequence[len(parsedSequence)-1][:-1]
    
    for i in range(len(parsedSequence)):
        parsedSequence[i] = float(parsedSequence[i])
    
    print("Fetching function")
    t1_start = perf_counter()
    resp = parallelComputing(parsedSequence, data, 5)
    t1_stop = perf_counter()
    print(resp)
    # print(jsonify(resp))
    # print("Elapsed time:", t1_stop - t1_start)
    return jsonify(resp)

@app.route('/')
def hello_world():
    return "<h1>Similar Search Algo</h1>"

if __name__ == '__main__':
    # print(get_vernacular_list(0))
    # app.run(host='127.0.0.2', port=5000)
    app.run(debug=True)

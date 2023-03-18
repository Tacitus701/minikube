import numpy as np
import json
import requests
from onnx import numpy_helper
from google.protobuf.json_format import MessageToJson
import base64

json_request_headers = {
'Content-Type': 'application/json',
'Accept': 'application/json'
}
pb_request_headers = {
'Content-Type': 'application/octet-stream',
'Accept': 'application/octet-stream'
}

input_array = np.array([[6.3, 3.3, 6.,  2.5]]).astype(np.float32)
tensor_proto = numpy_helper.from_array(input_array)

json_str = MessageToJson(tensor_proto, use_integers_for_enums=True)
data = json.loads(json_str)

inputs = {'float_input': data}
output_filters = ['probabilities']

payload = {"inputs": inputs, "outputFilter": output_filters}
ENDPOINT = 'http://localhost:80/v1/models/default/versions/1:predict'
res = requests.post(ENDPOINT, headers=json_request_headers, data=json.dumps(payload))

print(res.text)
raw_data = json.loads(res.text)['outputs']['probabilities']['rawData']
outputs = np.frombuffer(base64.b64decode(raw_data), dtype=np.float32)

# Get index of max value
print(np.argmax(outputs))
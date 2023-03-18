from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import numpy
import onnxruntime as rt


iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

clr = LogisticRegression()
clr.fit(X_train, y_train)
print(clr)

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type, options={id(clr): {'zipmap': False}})
"""
with open("logreg_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())
"""
sess = rt.InferenceSession(
    "logreg_iris.onnx", providers=rt.get_available_providers())
input_name = sess.get_inputs()[0].name
t = [[5.9, 3.,  4.2, 1.5],
 [6.2, 2.2, 4.5, 1.5],
 [6.3, 3.3, 6.,  2.5],
 [6.,  3.4, 4.5, 1.6],
 [5.6, 2.7, 4.2, 1.3]]
y = [1, 1, 2, 1, 1]

pred_onx = sess.run(None, {input_name: t})[0]
print(pred_onx)
print(accuracy_score(y, pred_onx))

for elt in sess.get_outputs():
    print(elt.name)

endpoint_names = ['image_tensor:0', 'output:0']
print(onx.graph.output)
for i in range(len(onx.graph.output)):
	if onx.graph.output[i].name in endpoint_names:
		print('-'*60)
		print(onx.graph.output[i])
		onx.graph.output[i].name = onx.graph.output[i].name.split(':')[0]
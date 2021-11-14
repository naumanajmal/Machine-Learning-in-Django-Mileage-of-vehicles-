from django.shortcuts import render
import joblib
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
reloadModel = joblib.load('./models/RFModelforMPG.pkl')

def index(request):
	return render(request, 'index.html')
def predictMPG(request):
	if request.method == 'POST':
		temp={}
		temp['cylinder'] = request.POST.get('cylinderval')
		temp['displacement'] = request.POST.get('disval')
		temp['horsepower'] = request.POST.get('hrspwrval')
		temp['weight'] = request.POST.get('weightval')
		temp['acceleration'] = request.POST.get('accval')
		temp['model_year'] = request.POST.get('modelval')
		temp['origin'] = request.POST.get('originval')
		temp2 = temp.copy()
		temp2['model year']=temp['model_year']
		del temp2['model_year']
		testdata = pd.DataFrame({'x':temp}).transpose()
		scoreval = reloadModel.predict(testdata)[0]
		context = {'scoreval': scoreval}
	return render(request, 'index.html', context)

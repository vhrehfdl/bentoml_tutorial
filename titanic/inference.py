import pickle

model_dir = "model_nb.pickle"

model = pickle.load(open(model_dir, 'rb'))
pred_y = model.predict([[3, 1, 17.4, 1, 0], [0, 0, 1.2, 1, 0]])
print(pred_y)
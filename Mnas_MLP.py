

from pandas import read_excel, DataFrame, concat
import numpy as np
from numpy import concatenate
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error




class MnasMLP:
    # --- Aktivasyon Fonksiyonları
    def sigmoid_act(self, x, der=False):
        if der:
            f = 1 / (1 + np.exp(-0.25 * x)) * (1 - 1 / (1 + np.exp(-0.25 * x)))
        else:
            f = 1 / (1 + np.exp(-0.25 * x))
        return f

    def tanh_act(self, x, der=False):
        if der:
            f = 1 - np.square(((np.exp(x)) - (np.exp(-x))) / ((np.exp(x)) + (np.exp(-x))))
        else:
            f = ((np.exp(x)) - (np.exp(-x))) / ((np.exp(x)) + (np.exp(-x)))
        return f

    def Lineer_act(self, x, der=False):
        if der:
            f = 1
        else:
            f = x
        return f

    # --- Katman Ekleme ve Güncelleme ---
    def add_layer(self, input_dim, output_dim):
        w_prev = np.zeros((output_dim, input_dim))
        w = 0.25 * np.random.randn(output_dim, input_dim)
        b = 0.25 * np.random.randn(output_dim)
        return {'w_prev': w_prev, 'w': w, 'b': b}

    def update_layer(self, layer, delta, input_vec, eta, alpha):
        w_new = layer['w'] + eta * np.outer(delta, input_vec) + alpha * (layer['w'] - layer['w_prev'])
        layer['w_prev'] = layer['w']
        layer['w'] = w_new
        layer['b'] = layer['b'] + eta * delta
        return layer

    # --- Feedforward ---
    def feed_forward(self, x, layers, act_fn, output_layer, output_act_fn):
        activations = [x]
        v_list = []
        for layer in layers:
            v = np.dot(layer['w'], activations[-1]) + layer['b']
            v_list.append(v)
            a = act_fn(v)
            activations.append(a)
        v_out = np.dot(output_layer['w'], activations[-1]) + output_layer['b']
        y_pred = output_act_fn(v_out)
        return activations, v_list, v_out, y_pred

    # --- Optimizer (Backpropagation) ---
    def optimizer(self, layers, output_layer, activations, v_list, v_out, y_true, act_fn, output_act_fn, eta, alpha):
        y_pred = output_act_fn(v_out)
        delta_out = (y_true - y_pred) * output_act_fn(v_out, der=True)
        output_layer = self.update_layer(output_layer, delta_out, activations[-1], eta, alpha)
        delta_next = delta_out
        w_next = output_layer['w']
        for i in reversed(range(len(layers))):
            v = v_list[i]
            a_prev = activations[i]
            delta = np.dot(delta_next, w_next) * act_fn(v, der=True)
            layers[i] = self.update_layer(layers[i], delta, a_prev, eta, alpha)
            delta_next = delta
            w_next = layers[i]['w']
        return output_layer, layers, y_pred

    # --- Egitim Fonksiyonu ---
    def train_MLP(self, train_X, train_y, hidden_sizes, n_steps, eta=1/300, alpha=1/900, epoch=1000,
                  act_fn=None, output_act_fn=None):
        if act_fn is None:
            act_fn = self.tanh_act
        if output_act_fn is None:
            output_act_fn = self.Lineer_act
        input_dim = train_X.shape[1]
        output_dim = n_steps
        layer_dims = [input_dim] + list(hidden_sizes)
        layers = []
        for i in range(len(hidden_sizes)):
            layers.append(self.add_layer(layer_dims[i], layer_dims[i+1]))
        output_layer = self.add_layer(hidden_sizes[-1], output_dim)
        E_ort = []
        E_ani_max = []
        for l in range(epoch):
            E_ani = []
            for k in range(train_X.shape[0]):
                x = train_X[k]
                y_true = train_y[k]
                activations, v_list, v_out, y_pred = self.feed_forward(x, layers, act_fn, output_layer, output_act_fn)
                output_layer, layers, y_pred = self.optimizer(
                    layers, output_layer, activations, v_list, v_out, y_true, act_fn, output_act_fn, eta, alpha
                )
                e = y_true - y_pred
                E_ani.append((1/2) * np.dot(e.T, e))
            E_ort.append((1 / train_X.shape[0]) * sum(E_ani))
            E_ani_max.append(max(E_ani))
            if l >= 21:
                if abs((E_ort[l-1]) - (E_ort[l])) <= 1e-10 or (E_ort[l-20]) - (E_ort[l]) < -0.0005:
                    print("E_ort_degisim=", (E_ort[l-20]) - (E_ort[l]))
                    break
        print("l=", l)
        return layers, output_layer, E_ort, E_ani_max

    # --- Tahmin Fonksiyonu ---
    def predict_MLP(self, X, layers, output_layer, act_fn, output_act_fn):
        preds = []
        for x in X:
            activations, v_list, v_out, y_pred = self.feed_forward(x, layers, act_fn, output_layer, output_act_fn)
            preds.append(y_pred)
        return np.array(preds)

# --- DATA HAZIRLIK ---

dataset = read_excel(r'C:\sxk96j_2_6ay.xlsx', header=0, index_col=0)
values = dataset.values
values = values.astype('float32')
n_hours = 6
n_steps = 3
n_features = 4
n_obs = n_hours * n_features

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

reframed = series_to_supervised(values, n_hours, n_steps)
for i in range (0,n_steps):
    reframed.drop(reframed.columns[[n_obs+i,n_obs+i+1,n_obs+i+3]], axis=1, inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
reframed = scaler.fit_transform(reframed)

TRAIN_SIZE = 0.60
train = reframed[:int(len(reframed)*TRAIN_SIZE), :]
test = reframed[int(len(reframed)*TRAIN_SIZE):, :]
train_X, train_y = train[:, :n_obs], train[:, -n_steps:]
test_X, test_y = test[:, :n_obs], test[:, -n_steps:]
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape, len(test_X))

# --- MLP Eğitim ---
hidden_sizes = [30, 20, 20, 30]  # Katman nöron sayıları
mlp = MnasMLP()
layers, output_layer, E_ort, E_ani_max = mlp.train_MLP(
    train_X, train_y, hidden_sizes, n_steps, 
    eta=1/300, alpha=1/900, epoch=1000, act_fn=mlp.tanh_act, output_act_fn=mlp.Lineer_act
)

# --- Tahmin ---
pred_train_y = mlp.predict_MLP(train_X, layers, output_layer, mlp.tanh_act, mlp.Lineer_act)
pred_test_y = mlp.predict_MLP(test_X, layers, output_layer, mlp.tanh_act, mlp.Lineer_act)

# --- Ölçekleri geri çevirme ---
pred_train_y = concatenate((train_X[:, -n_obs:], pred_train_y), axis=1)
pred_train_y = scaler.inverse_transform(pred_train_y)
pred_train_y = pred_train_y[:,-n_steps:]

train_y_inv = concatenate((train_X[:, -n_obs:], train_y), axis=1)
train_y_inv = scaler.inverse_transform(train_y_inv)
train_y_inv = train_y_inv[:,-n_steps:]

pred_test_y = concatenate(( test_X[:, -n_obs:], pred_test_y), axis=1)
pred_test_y = scaler.inverse_transform(pred_test_y)
pred_test_y = pred_test_y[:,-n_steps:]

test_y_inv = concatenate((test_X[:, -n_obs:], test_y), axis=1)
test_y_inv = scaler.inverse_transform(test_y_inv)
test_y_inv = test_y_inv[:,-n_steps:]


#%%


# --- Hata ve Başarı Metrikleri ---
rmse_train = sqrt(mean_squared_error(train_y_inv, pred_train_y))
print("Train data score: %.2f RMSE" % rmse_train)
rmse_test = sqrt(mean_squared_error(test_y_inv, pred_test_y))
print("Test data score: %.2f RMSE" % rmse_test)

from sklearn.metrics import mean_absolute_percentage_error
error_mape_tr = 100*mean_absolute_percentage_error(train_y_inv, pred_train_y)
print("Train data score: %.2f MAPE" % error_mape_tr)
error_mape_t = 100*mean_absolute_percentage_error(test_y_inv, pred_test_y)
print("Test data score: %.2f MAPE" % error_mape_t)
error_mape_t0 = 100*mean_absolute_percentage_error(test_y_inv[:,0], pred_test_y[:,0])
print("Test t zamanı score: %.2f MAPE" % error_mape_t0)
error_mape_t1 = 100*mean_absolute_percentage_error(test_y_inv[:,1], pred_test_y[:,1])
print("Test t+1 zamanı score: %.2f MAPE" % error_mape_t1)
error_mape_t2 = 100*mean_absolute_percentage_error(test_y_inv[:,2], pred_test_y[:,2])
print("Test t+2 zamanı score: %.2f MAPE" % error_mape_t2)

mdape_tr = np.median((np.abs(np.subtract(train_y_inv, pred_train_y)/ train_y_inv))) * 100
print("Train data score: %.2f MdAPE" % mdape_tr)
mdape_t = np.median((np.abs(np.subtract(test_y_inv, pred_test_y)/ test_y_inv))) * 100
print("Test data score: %.2f MdAPE" % mdape_t)

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

error_smape_tr = smape(train_y_inv, pred_train_y)
print("Train data score: %.2f SMAPE" % error_smape_tr)
error_smape_t = smape(test_y_inv, pred_test_y)
print("Test data score: %.2f SMAPE" % np.mean(error_smape_t))

from sklearn.metrics import mean_absolute_error
e_tr = train_y_inv - pred_train_y
scale = mean_absolute_error(train_y_inv[1:], train_y_inv[:-1])
mase_tr = np.mean(np.abs(e_tr / scale))
print("Train data score: %.2f MASE" % mase_tr)
e_t = test_y_inv - pred_test_y
scale = mean_absolute_error(test_y_inv[1:], test_y_inv[:-1])
mase_t = np.mean(np.abs(e_t / scale))
print("Test data score: %.2f MASE" % mase_t)



# --- Grafikler ---
plt.figure(figsize=(12, 4))
plt.scatter(np.arange(0, len(E_ort)), E_ort, alpha=0.5, s=10, label='Error')
plt.plot(E_ort)
plt.title('Loss for each training data point', fontsize=20)
plt.xlabel('Training data', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.show()

# t zamanı
train_y1 = train_y_inv[:,0]
pred_train_y1 = pred_train_y[:,0]
plt.figure(figsize=(48, 16))
plt.plot(pred_train_y1, label="Train verisi t zamanı Tahmin")
plt.plot(train_y1, label="Train verisi t zamanı Gerçek")
plt.title('Train verisi t zamanı Tahmin ve Gercek zamanla degisimi', fontsize=32)
plt.xlabel('Zaman', fontsize=32)
plt.ylabel('Hız Değerleri', fontsize=32)
plt.legend(fontsize=32)
plt.savefig('MLP_Mnas_15k_iter_train_t.png')

test_y1 = test_y_inv[:,0]
pred_test_y1 = pred_test_y[:,0]
plt.figure(figsize=(48, 16))
plt.plot(pred_test_y1, label="Test verisi t zamanı Tahmin")
plt.plot(test_y1, label="Test verisi t zamanı Gerçek")
plt.title('Test verisi t zamanı Tahmin ve Gercek zamanla degisimi', fontsize=32)
plt.xlabel('Zaman', fontsize=32)
plt.ylabel('Hız Değerleri', fontsize=32)
plt.legend(fontsize=32)
plt.savefig('MLP_Mnas_15k_iter_test_t.png')

# t+1 zamanı
train_y2 = train_y_inv[:,1]
pred_train_y2 = pred_train_y[:,1]
plt.figure(figsize=(48, 16))
plt.plot(pred_train_y2, label="Train verisi t+1 zamanı Tahmin")
plt.plot(train_y2, label="Train verisi t+1 zamanı Gerçek")
plt.title('Train verisi t+1 zamanı Tahmin ve Gercek zamanla degisimi', fontsize=32)
plt.xlabel('Zaman', fontsize=32)
plt.ylabel('Hız Değerleri', fontsize=32)
plt.legend(fontsize=32)
plt.savefig('MLP_Mnas_15k_iter_train_t+1.png')

test_y2 = test_y_inv[:,1]
pred_test_y2 = pred_test_y[:,1]
plt.figure(figsize=(48, 16))
plt.plot(pred_test_y2, label="Test verisi t+1 zamanı Tahmin")
plt.plot(test_y2, label="Test verisi t+1 zamanı Gerçek")
plt.title('Test verisi t+1 zamanı Tahmin ve Gercek zamanla degisimi', fontsize=32)
plt.xlabel('Zaman', fontsize=32)
plt.ylabel('Hız Değerleri', fontsize=32)
plt.legend(fontsize=32)
plt.savefig('MLP_Mnas_15k_iter_test_t+1.png')

# t+2 zamanı
train_y3 = train_y_inv[:,2]
pred_train_y3 = pred_train_y[:,2]
plt.figure(figsize=(48, 16))
plt.plot(pred_train_y3, label="Train verisi t+2 zamanı Tahmin")
plt.plot(train_y3, label="Train verisi t+2 zamanı Gerçek")
plt.title('Train verisi t+2 zamanı Tahmin ve Gercek zamanla degisimi', fontsize=32)
plt.xlabel('Zaman', fontsize=32)
plt.ylabel('Hız Değerleri', fontsize=32)
plt.legend(fontsize=32)
plt.savefig('MLP_Mnas_15k_iter_train_t+2.png')

test_y3 = test_y_inv[:,2]
pred_test_y3 = pred_test_y[:,2]
plt.figure(figsize=(48, 16))
plt.plot(pred_test_y3, label="Test verisi t+2 zamanı Tahmin")
plt.plot(test_y3, label="Test verisi t+2 zamanı Gerçek")
plt.title('Test verisi t+2 zamanı Tahmin ve Gercek zamanla degisimi', fontsize=32)
plt.xlabel('Zaman', fontsize=32)
plt.ylabel('Hız Değerleri', fontsize=32)
plt.legend(fontsize=32)
plt.savefig('MLP_Mnas_15k_iter_test_t+2.png')

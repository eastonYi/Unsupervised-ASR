import tensorflow as tf


def get_LSTM_UWb(weight):
    '''
    weight must be output of LSTM's layer.get_weights()
    W: weights for input
    U: weights for hidden states
    b: bias
    '''
    warr, uarr, barr = weight
    gates = ["i","f","c","o"]
    hunit = uarr.shape[0]
    U, W, b = {},{},{}
    for i1,i2 in enumerate(range(0,len(barr),hunit)):

        W[gates[i1]] = warr[:,i2:i2+hunit]
        U[gates[i1]] = uarr[:,i2:i2+hunit]
        b[gates[i1]] = barr[i2:i2+hunit].reshape(hunit,1)

    return W, U, b


def get_GRU_UWb(weight):
    '''
    weight must be output of GRU's layer.get_weights()
    W: weights for input
    U: weights for hidden states
    b: bias
    '''
    warr, uarr, barr = weight
    gates = ["r","u","o"]
    hunit = uarr.shape[0]
    U, W, b = {}, {}, {}
    for i1, i2 in enumerate(range(0, len(barr), hunit)):
        W[gates[i1]] = warr[:,i2:i2+hunit]
        U[gates[i1]] = uarr[:,i2:i2+hunit]
        b[gates[i1]] = barr[i2:i2+hunit].reshape(hunit,1)

    return W, U, b


def get_LSTMweights(model1):
    for layer in model1.layers:
        if "LSTM" in str(layer):
            w = layer.get_weights()
            W,U,b = get_LSTM_UWb(w)
            break
    return W, U, b


def get_GRUweights(model):
    for layer in model.layers:
        if "GRU" in str(layer):
            w = layer.get_weights()
            W,U,b = get_GRU_UWb(w)
            break

    return W, U, b


def vectorize_with_labels(W,U,b):
    bs,bs_label,ws,ws_label,us,us_label=[],[],[],[],[],[]
    for k in ["i","f","c","o"]:
        temp = list(W[k].flatten())
        ws_label.extend(["W_"+k]*len(temp))
        ws.extend(temp)

        temp = list(U[k].flatten())
        us_label.extend(["U_"+k]*len(temp))
        us.extend(temp)

        temp = list(b[k].flatten())
        bs_label.extend(["b_"+k]*len(temp))
        bs.extend(temp)
    weight = ws + us + bs
    wlabel = ws_label + us_label + bs_label
    return(weight,wlabel)

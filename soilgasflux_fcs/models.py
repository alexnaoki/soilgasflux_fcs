

def hm_model(t, cx, a, t0, c0):
    e = 2.71828
    return cx+(c0-cx)*e**(-a*(t-t0))

def hm_model_dcdt(t0, c0, a, cx, t):
    e = 2.71828
    dcdt = a*(cx - c0)*e**(-a*(t-t0))
    return dcdt

def linear_model(t, dcdt, c0):
    '''
    linear function

    y = dcdt * t + c0
    '''
    return dcdt*t + c0

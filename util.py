import numpy as np

class lookup_1D:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def index(self,xq):
        return np.interp(xq, self.x, self.y)
    
    def index_y(self,yq):
        return np.interp(yq, self.y, self.x)
    
def rec_prism_inertia(x_m,y_m,z_m,m_kg):
    Ixx_kgm2 = 1/12 * m_kg * (y_m**2 + z_m**2)
    Iyy_kgm2 = 1/12 * m_kg * (x_m**2 + z_m**2)
    Izz_kgm2 = 1/12 * m_kg * (x_m**2 + y_m**2)
    return np.diag([Ixx_kgm2,Iyy_kgm2,Izz_kgm2])

def mm2m(dim_mm):
    return dim_mm/1000

def g2kg(mass_g):
    return mass_g/1000

def ms2s(time_ms):
    return time_ms/1000

def rad2deg(val_rad):
    return val_rad * 180 / np.pi

def deg2rad(val_deg):
    return val_deg * np.pi / 180

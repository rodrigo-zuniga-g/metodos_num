"""Propiedades de una viga de acero IPE 300 (S275) expuesta a un incendio localizado.

MEC301 - Métodos Numéricos, Tarea 3 (2 sem 2026).

Constantes del módulo (se leen como vf.L, vf.h, etc., después de `import viga_fuego as vf`)
-------------------------------------------------------------------------------------------
L      largo de la viga, 6.0 m
h      altura de la sección IPE 300, 0.300 m
I      momento de inercia de la sección IPE 300, 8.356e-5 m^4 (no cambia con la temperatura)
E20    módulo de elasticidad del acero a 20 °C, 210e9 Pa
fy20   límite de fluencia del acero S275 a 20 °C, 275e6 Pa
T_amb, T_max, x_fuego, ancho_fuego   parámetros del perfil de temperatura (°C, °C, m, m)

Funciones
---------
T_x(x)          temperatura del acero (°C) en la posición x (m)
E_T(T, eps=0)   módulo de elasticidad (Pa) del acero a la temperatura T (°C) y deformación eps

Modelo del material: EN 1993-1-2 (Eurocódigo 3, parte 1-2), Tabla 3.1 y Figura 3.1, con
interpolación lineal en T. Para eps = 0 (o una deformación dentro del rango proporcional),
E_T entrega el módulo elástico E(T). Para deformaciones mayores, entrega el módulo secante
sigma(eps, T)/eps, que disminuye a medida que el acero se acerca a la fluencia.
"""
import numpy as np

# ---- Constantes --------------------------------------------------------------
L = 6.0                  # m
h = 0.300                # m
I = 8.356e-5             # m^4, IPE 300
E20 = 210e9              # Pa
fy20 = 275e6             # Pa

T_amb, T_max, x_fuego, ancho_fuego = 20.0, 650.0, 3.5, 1.2

# ---- EN 1993-1-2, Tabla 3.1 (acero al carbono) --------------------------------
_T  = np.array([20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200.])
_ky = np.array([1, 1, 1, 1, 1, .78, .47, .23, .11, .06, .04, .02, 0.])
_kp = np.array([1, 1, .807, .613, .420, .360, .180, .075, .050, .0375, .025, .0125, 0.])
_kE = np.array([1, 1, .9, .8, .7, .6, .31, .13, .09, .0675, .045, .0225, 0.])


def T_x(x):
    """Temperatura del acero (°C) en la posición x (m). Acepta escalares o arreglos."""
    return T_amb + (T_max - T_amb)*np.exp(-((np.asarray(x) - x_fuego)/ancho_fuego)**2)


def _sigma(eps, T):
    """Tensión (Pa) para la deformación eps >= 0, EN 1993-1-2 Fig. 3.1 (sin rama descendente)."""
    E  = np.interp(T, _T, _kE)*E20
    fp = np.interp(T, _T, _kp)*fy20
    fy = np.interp(T, _T, _ky)*fy20
    ep, ey = fp/E, 0.02
    c = (fy - fp)**2/((ey - ep)*E - 2*(fy - fp))
    a = np.sqrt((ey - ep)*(ey - ep + c/E))
    b = np.sqrt(c*(ey - ep)*E + c**2)
    return np.where(eps <= ep, E*eps,
           np.where(eps < ey, fp - c + b/a*np.sqrt(np.maximum(a**2 - (ey - eps)**2, 0.0)), fy))


def E_T(T, eps=0.0):
    """
    Módulo de elasticidad del acero (Pa) a la temperatura T y la deformación eps.

    Parameters
    ----------
    T : float o ndarray
        Temperatura del acero (°C).
    eps : float o ndarray, opcional
        Deformación unitaria (adimensional). Se usa su valor absoluto. Por defecto 0, que
        entrega el módulo elástico E(T).

    Returns
    -------
    E : float o ndarray
        Módulo (Pa). Es el módulo elástico E(T) mientras eps está en el rango proporcional,
        y el módulo secante sigma(eps, T)/eps para deformaciones mayores.
    """
    T = np.asarray(T, float)
    e = np.abs(np.asarray(eps, float))
    E0 = np.interp(T, _T, _kE)*E20
    e_safe = np.where(e > 0, e, 1.0)
    Es = np.where(e > 0, _sigma(e_safe, T)/e_safe, E0)
    return float(Es) if Es.ndim == 0 else Es

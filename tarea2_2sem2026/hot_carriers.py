"""
hot_carriers.py
Funciones auxiliares para el Problema 1 de la Tarea 2 de MEC301 - Métodos Numéricos (2 sem 2026).

Este archivo arma el cálculo de la distribución de portadores calientes (hot carriers)
a partir de DOS funciones que usted programa en la tarea:

    raices_jl(l, x_max)                 -> ceros de la función de Bessel esférica j_l en (0, x_max]
    matriz_transicion(li, xi, lf, xf)   -> matriz |M_fi|^2 entre estados iniciales y finales

Uso típico:

    from hot_carriers import distribucion_hot_carriers, graficar_distribucion
    res = distribucion_hot_carriers(raices_jl, matriz_transicion, D=15, hw=3.65,
                                    tau_list=[0.05, 0.1, 0.5, 1.0])
    graficar_distribucion(res)

No es necesario (ni se pide) modificar este archivo.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

C_HBAR2_2M = 0.0380998      # hbar^2/(2 m_e), en eV nm^2
HBAR_EVPS  = 6.582119569E-4 # constante de Planck reducida, en eV ps
EF_AG      = 5.49           # energía de Fermi de la plata, en eV


def energia(x, D):
    '''
    Energía (eV) de un estado del pozo esférico infinito de diámetro D (nm),
    a partir del cero x de la función de Bessel esférica: E = hbar^2 x^2/(2 m a^2).
    '''
    a = D/2
    return C_HBAR2_2M*(np.asarray(x)/a)**2


def x_de_energia(E, D):
    '''
    Inverso de `energia`: valor de x = k a asociado a la energía E (eV).
    '''
    a = D/2
    return a*np.sqrt(E/C_HBAR2_2M)


def estados_pozo(raices_jl, D, E_max):
    '''
    Construye la lista de todos los niveles (n, l) del pozo esférico infinito de
    diámetro D (nm) con energía menor o igual a E_max (eV), usando la función
    raices_jl(l, x_max) del alumno.

    Return
    -------
        - l : ndarray (int)   número cuántico angular de cada nivel
        - x : ndarray         cero de j_l asociado (x = k a)
        - E : ndarray         energía del nivel, en eV
    '''
    x_max = x_de_energia(E_max, D)
    L, X = [], []
    l = 0
    while True:
        r = np.atleast_1d(np.asarray(raices_jl(l, x_max), dtype=float))
        r = r[(r > 1E-8) & (r <= x_max)]
        if len(r) == 0:
            break
        L += [l]*len(r)
        X += list(r)
        l += 1
    L, X = np.array(L, dtype=int), np.array(X)
    orden = np.argsort(X)
    L, X = L[orden], X[orden]
    return L, X, energia(X, D)


def _lorentz(e, gamma):
    return 1.0/(e**2 + gamma**2)


def suavizar(E, G, E_grilla, ancho):
    '''
    Convierte una tasa discreta G(E_k), definida nivel por nivel, en una curva
    continua sobre la grilla E_grilla. Cada nivel se reemplaza por una lorentziana
    de área unitaria y ancho completo a media altura `ancho` (eV):

        G(E) = sum_k G_k L(E - E_k),   L(x) = (1/pi) (ancho/2) / (x^2 + (ancho/2)^2)

    Es el mismo procedimiento que usa la librería plytrons (convert_raw_hot) para
    graficar distribuciones como las de Manjavacas et al. (2014). El área bajo la
    curva es igual a la suma de las tasas, así que la curva queda en unidades de
    G por eV.
    '''
    g = ancho/2
    K = (g/np.pi)/((E_grilla[:, None] - np.asarray(E)[None, :])**2 + g**2)
    return K @ np.asarray(G)


def distribucion_hot_carriers(raices_jl, matriz_transicion, D=15.0, hw=3.65,
                              tau_list=(0.05, 0.1, 0.5, 1.0), EF=EF_AG,
                              margen=0.3, n_grilla=4000, dE_factor=5, verbose=True):
    '''
    Calcula la tasa de generación de electrones y huecos calientes (regla de oro
    de Fermi, ec. 1 de Manjavacas et al., ACS Nano 8, 7630, 2014) para una
    nanopartícula de diámetro D, iluminada a la energía hw.

    Parametros
    ------------
        - raices_jl : callable
                función del alumno, raices_jl(l, x_max)
        - matriz_transicion : callable
                función del alumno, matriz_transicion(li, xi, lf, xf) -> (Ni, Nf)
        - D : float
                diámetro de la nanopartícula, en nm (15 por defecto)
        - hw : float
                energía del fotón, en eV (3.65 por defecto, el plasmón de Ag)
        - tau_list : lista de float
                tiempos de vida de los portadores, en ps
        - EF : float
                energía de Fermi, en eV (5.49 por defecto, plata)
        - dE_factor : float
                ancho del suavizado de las curvas, en múltiplos del paso de una
                grilla de 1000 puntos entre el fondo del pozo y EF + hw (5 por
                defecto, como en plytrons). No afecta 'Gamma_e' ni 'Gamma_h'.

    Return
    -------
        - res : dict con las llaves
            'D', 'hw', 'tau', 'energia' (grilla, eV respecto de EF),
            'electrones', 'huecos' (una curva por cada tau, en portadores por eV
            de energía absorbida y por eV de energía del portador),
            'E_f', 'E_i' (energía de cada nivel final e inicial, eV respecto de EF),
            'Gamma_e', 'Gamma_h' (tasa por nivel, sin ensanchar, una por cada tau),
            'n_niveles', 'n_transiciones', 't_matriz' (segundos)
    '''
    # ---- niveles en la ventana de energía relevante ----------------------------
    l, x, E = estados_pozo(raices_jl, D, EF + hw + margen)
    ini = (E <= EF) & (E >= EF - hw - margen)     # ocupados (huecos)
    fin = (E > EF)                                # vacíos (electrones)

    # ---- elementos de matriz (función del alumno) -------------------------------
    tic = time.perf_counter()
    M2 = np.asarray(matriz_transicion(l[ini], x[ini], l[fin], x[fin]), dtype=float)
    t_matriz = time.perf_counter() - tic
    assert M2.shape == (ini.sum(), fin.sum()), \
        f'matriz_transicion debe retornar un arreglo de {ini.sum()} x {fin.sum()}'

    dl = np.abs(l[fin][None, :] - l[ini][:, None])
    n_trans = int((dl == 1).sum())
    if verbose:
        print(f'D = {D} nm: {len(E)} niveles hasta EF + hw, '
              f'{ini.sum()} iniciales x {fin.sum()} finales, '
              f'{n_trans} transiciones permitidas, matriz en {t_matriz:.2f} s')

    Ei, Ef = E[ini], E[fin]
    dE = Ef[None, :] - Ei[:, None]                # siempre positivo
    eg = np.linspace(-hw - margen, hw + margen, n_grilla)
    # ancho del suavizado (FWHM): dE_factor veces el paso de una grilla de 1000
    # puntos que cubre todos los niveles, desde el fondo del pozo hasta EF + hw
    ancho = dE_factor*(EF + hw + margen)/999

    res = {'D': D, 'hw': hw, 'tau': list(tau_list), 'energia': eg,
           'electrones': [], 'huecos': [], 'n_niveles': len(E), 'ancho_suavizado': ancho,
           'n_transiciones': n_trans, 't_matriz': t_matriz,
           'E_f': Ef - EF, 'E_i': Ei - EF, 'Gamma_e': [], 'Gamma_h': []}

    for tau in tau_list:
        gam = HBAR_EVPS/tau                       # ancho de nivel hbar/tau, eV
        G = M2*(_lorentz(hw - dE, gam) + _lorentz(hw + dE, gam))
        # conservación de energía: la potencia absorbida es la misma para todo tau,
        # así que normalizamos por la energía total entregada a los portadores
        G = G/np.sum(dE*G)
        Ge = G.sum(axis=0)                        # por nivel final (electrones)
        Gh = G.sum(axis=1)                        # por nivel inicial (huecos)
        # cada nivel se reemplaza por una lorentziana de ancho fijo (suavizado)
        res['Gamma_e'].append(Ge)
        res['Gamma_h'].append(Gh)
        res['electrones'].append(suavizar(Ef - EF, Ge, eg, ancho))
        res['huecos'].append(suavizar(Ei - EF, Gh, eg, ancho))
    return res


def graficar_distribucion(res, escala_log=False):
    '''
    Grafica la distribución de electrones calientes (rojo, energía > 0) y de huecos
    calientes (azul, energía < 0) calculada con distribucion_hot_carriers, con el
    estilo de Manjavacas et al. (2014): curvas suavizadas y área rellena. Se dibuja
    un panel por cada tiempo de vida tau; las barras verticales marcan +-hw.

    Parametros
    ------------
        - res : dict
                resultado de distribucion_hot_carriers
        - escala_log : bool
                eje vertical logarítmico (False por defecto)
    Return
    -------
        - fig, axs : figura y ejes de matplotlib
    '''
    eg = res['energia']
    hw = res['hw']
    ntau = len(res['tau'])
    fig, axs = plt.subplots(ntau, 1, figsize=(6.5, 1.5*ntau + 0.8), sharex=True)
    axs = np.atleast_1d(axs)
    me, mh = eg >= 0, eg <= 0
    for k, tau in enumerate(res['tau']):
        ax = axs[k]
        e, h = res['electrones'][k], res['huecos'][k]
        ax.fill_between(eg[me], e[me], color='tab:red', alpha=0.35, lw=0)
        ax.plot(eg[me], e[me], color='tab:red', lw=0.9, label='electrones')
        ax.fill_between(eg[mh], h[mh], color='tab:blue', alpha=0.35, lw=0)
        ax.plot(eg[mh], h[mh], color='tab:blue', lw=0.9, label='huecos')
        ax.axvline(0, color='k', lw=0.6, ls='--')
        for s_ in (-1, 1):
            ax.axvline(s_*hw, color='gray', lw=0.6, ls=':')
        ax.text(0.99, 0.80, f'$\\tau$ = {tau} ps', transform=ax.transAxes, ha='right', fontsize=9)
        if escala_log:
            ax.set_yscale('log')
        else:
            ax.set_ylim(0, None)
        ax.grid(alpha=0.3)
    axs[0].set_title(f"D = {res['D']} nm, $\\hbar\\omega$ = {hw} eV", fontsize=10)
    axs[0].legend(fontsize=8, loc='upper left')
    axs[-1].set_xlabel(r'energía respecto del nivel de Fermi, $\varepsilon - E_F$ (eV)')
    axs[-1].set_xlim(eg[0], eg[-1])
    fig.supylabel('portadores / (eV absorbido $\\cdot$ eV)', fontsize=9)
    fig.tight_layout()
    return fig, axs

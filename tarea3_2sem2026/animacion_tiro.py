"""Animación interactiva de un tiro libre (MEC301, Tarea 3).

Sistema de coordenadas: x apunta del punto de tiro hacia el arco, y es lateral
(positivo hacia la izquierda del pateador) y z es vertical. El balón parte en el origen.

Uso básico
----------
>>> import animacion_tiro as at
>>> fig = at.animar_tiro(sol.t, sol.y[0], sol.y[1], sol.y[2])
>>> fig.show()                       # en Jupyter
>>> fig.write_html('tiro.html')      # archivo independiente

Controles: ▶ reproducir, ⏸ pausar, ⏮ volver al inicio, barra de tiempo para avanzar o
retroceder cuadro a cuadro, y botones de cámara (pateador, arquero, superior, lateral, 3D).
La vista 3D también se puede rotar y acercar con el mouse.

Requiere plotly (pip install plotly).
"""
import numpy as np
import plotly.graph_objects as go

ANCHO_ARCO = 7.32     # m (Reglas de Juego IFAB)
ALTO_ARCO = 2.44      # m
DIST_BARRERA = 9.15   # m
FONDO_RED = 2.0       # m, profundidad de la red (supuesta)

CAMARAS = {
    'Pateador': dict(eye=dict(x=-1.1, y=0.0, z=0.25), center=dict(x=0.1, y=0, z=-0.1), up=dict(x=0, y=0, z=1)),
    'Arquero':  dict(eye=dict(x=1.05, y=0.0, z=0.12), center=dict(x=0.45, y=0, z=-0.05), up=dict(x=0, y=0, z=1)),
    'Superior': dict(eye=dict(x=0.0, y=0.0, z=2.2), center=dict(x=0, y=0, z=0), up=dict(x=1, y=0, z=0)),
    'Lateral':  dict(eye=dict(x=0.0, y=-2.2, z=0.1), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
    '3D':       dict(eye=dict(x=-1.2, y=-1.4, z=0.8), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
}


def _lineas(segmentos):
    """Convierte una lista de segmentos [(p0, p1), ...] en arreglos x, y, z separados por None."""
    xs, ys, zs = [], [], []
    for p0, p1 in segmentos:
        xs += [p0[0], p1[0], None]; ys += [p0[1], p1[1], None]; zs += [p0[2], p1[2], None]
    return xs, ys, zs


def _escena_fija(d, y_arco, barrera, y_lim):
    """Trazas que no cambian: cancha, arco, red y barrera."""
    trazas = []
    a, h = ANCHO_ARCO/2, ALTO_ARCO
    yl, yr = y_arco + a, y_arco - a
    # cancha
    trazas.append(go.Mesh3d(x=[-2, d + FONDO_RED + 1, d + FONDO_RED + 1, -2],
                            y=[y_lim[0], y_lim[0], y_lim[1], y_lim[1]], z=[0, 0, 0, 0],
                            i=[0, 0], j=[1, 2], k=[2, 3], color='#3f8f3f', opacity=0.35,
                            hoverinfo='skip', showlegend=False))
    lin = [((d, y_lim[0], 0), (d, y_lim[1], 0)),                   # línea de gol
           ((d - 16.5, yr - 16.5 + a, 0), (d - 16.5, yl + 16.5 - a, 0)),  # borde del área grande
           ((d - 16.5, yr - 16.5 + a, 0), (d, yr - 16.5 + a, 0)),
           ((d - 16.5, yl + 16.5 - a, 0), (d, yl + 16.5 - a, 0))]
    x, y, z = _lineas(lin)
    trazas.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='white', width=3),
                               hoverinfo='skip', name='cancha', showlegend=False))
    # red: fondo y laterales
    red = []
    for yy in np.linspace(yr, yl, 13):
        red += [((d + FONDO_RED, yy, 0), (d + FONDO_RED, yy, h)), ((d, yy, h), (d + FONDO_RED, yy, h))]
    for zz in np.linspace(0, h, 6):
        red += [((d + FONDO_RED, yr, zz), (d + FONDO_RED, yl, zz)),
                ((d, yr, zz), (d + FONDO_RED, yr, zz)), ((d, yl, zz), (d + FONDO_RED, yl, zz))]
    x, y, z = _lineas(red)
    trazas.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='lightgray', width=1),
                               hoverinfo='skip', name='red', showlegend=False))
    # postes y travesaño
    x, y, z = _lineas([((d, yr, 0), (d, yr, h)), ((d, yl, 0), (d, yl, h)), ((d, yr, h), (d, yl, h))])
    trazas.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='white', width=9),
                               name='arco', hoverinfo='skip', showlegend=False))
    # barrera: un segmento vertical grueso por jugador
    if barrera is not None:
        xb, yb, hb = barrera.get('x', DIST_BARRERA), barrera['y'], barrera.get('altura', 1.85)
        x, y, z = _lineas([((xb, yy, 0), (xb, yy, hb)) for yy in yb])
        trazas.append(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='#1f3b8f', width=18),
                                   name='barrera', hoverinfo='skip'))
    return trazas


def resultado_tiro(t, x, y, z, d=35.0, y_arco=0.0):
    """Indica si el balón cruza la línea de gol dentro del arco.

    Returns
    -------
    gol : bool
    cruce : tuple o None
        (t, y, z) en el cruce de x = d, interpolado linealmente; None si no llega.
    """
    x = np.asarray(x)
    k = np.nonzero((x[:-1] < d) & (x[1:] >= d))[0]
    if k.size == 0:
        return False, None
    k = k[0]
    s = (d - x[k])/(x[k+1] - x[k])
    tc, yc, zc = [(1 - s)*q[k] + s*q[k+1] for q in (t, y, z)]
    gol = abs(yc - y_arco) < ANCHO_ARCO/2 and 0 <= zc < ALTO_ARCO
    return bool(gol), (tc, yc, zc)


def animar_tiro(t, x, y, z, d=35.0, y_arco=0.0, barrera=None, dt=0.02,
                vista='Pateador', titulo='Tiro libre'):
    """Animación 3D interactiva de la trayectoria del balón.

    Parameters
    ----------
    t, x, y, z : array_like
        Tiempo (s) y posición del balón (m), por ejemplo sol.t y sol.y[0:3] de solve_ivp.
        No hace falta que t esté equiespaciado: la función interpola linealmente cada dt.
    d : float
        Distancia del punto de tiro a la línea de gol (m).
    y_arco : float
        Posición lateral del centro del arco (m).
    barrera : dict, opcional
        {'x': distancia (m), 'y': lista de posiciones laterales de los jugadores (m),
         'altura': m}. Si es None no se dibuja.
    dt : float
        Intervalo entre cuadros de la animación (s).
    vista : str
        Cámara inicial: 'Pateador', 'Arquero', 'Superior', 'Lateral' o '3D'.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    t, x, y, z = (np.asarray(q, float) for q in (t, x, y, z))
    tf = np.arange(t[0], t[-1] + 0.5*dt, dt)
    xf, yf, zf = (np.interp(tf, t, q) for q in (x, y, z))
    gol, cruce = resultado_tiro(t, x, y, z, d, y_arco)

    y_lim = (min(yf.min(), y_arco - ANCHO_ARCO) - 3, max(yf.max(), y_arco + ANCHO_ARCO) + 3)
    fijas = _escena_fija(d, y_arco, barrera, y_lim)
    n_fijas = len(fijas)

    def dinamicas(i):
        return [go.Scatter3d(x=xf[:i+1], y=yf[:i+1], z=zf[:i+1], mode='lines',
                             line=dict(color='orange', width=5), name='trayectoria'),
                go.Scatter3d(x=[xf[i]], y=[yf[i]], z=[zf[i]], mode='markers',
                             marker=dict(size=7, color='white', line=dict(color='black', width=2)),
                             name='balón'),
                go.Scatter3d(x=[xf[i]], y=[yf[i]], z=[0], mode='markers',
                             marker=dict(size=5, color='black', opacity=0.35), name='sombra',
                             showlegend=False)]

    def texto(i):
        v = np.sqrt(np.gradient(xf, tf)[i]**2 + np.gradient(yf, tf)[i]**2 + np.gradient(zf, tf)[i]**2)
        s = f't = {tf[i]:.2f} s<br>|V| ≈ {v:.1f} m/s<br>z = {zf[i]:.2f} m'
        if i == len(tf) - 1 and cruce is not None:
            s += f"<br><b>{'GOL' if gol else 'FUERA'}</b> (y = {cruce[1]:.2f} m, z = {cruce[2]:.2f} m)"
        return [dict(text=s, x=0.01, y=0.97, xref='paper', yref='paper', showarrow=False,
                     align='left', bgcolor='rgba(255,255,255,0.8)', font=dict(size=13))]

    frames = [go.Frame(data=dinamicas(i), traces=list(range(n_fijas, n_fijas + 3)), name=str(i),
                       layout=go.Layout(annotations=texto(i))) for i in range(len(tf))]

    rango = dict(x=[-2, d + FONDO_RED + 1], y=list(y_lim), z=[0, max(zf.max() + 1, ALTO_ARCO + 2)])
    escala = np.array([rango['x'][1] - rango['x'][0], y_lim[1] - y_lim[0], rango['z'][1]])
    escala = escala/escala.max()

    play = dict(frame=dict(duration=1000*dt, redraw=True), fromcurrent=True, transition=dict(duration=0))
    quieto = dict(frame=dict(duration=0, redraw=True), mode='immediate', transition=dict(duration=0))
    fig = go.Figure(data=fijas + dinamicas(0), frames=frames)
    fig.update_layout(
        title=dict(text=titulo, y=0.98), height=650, margin=dict(l=0, r=0, t=70, b=40), annotations=texto(0),
        scene=dict(xaxis=dict(range=rango['x'], title='x (m)'),
                   yaxis=dict(range=rango['y'], title='y (m)'),
                   zaxis=dict(range=rango['z'], title='z (m)'),
                   aspectmode='manual', aspectratio=dict(x=escala[0], y=escala[1], z=escala[2]),
                   camera=CAMARAS[vista]),
        updatemenus=[
            dict(type='buttons', direction='left', x=0.0, y=0.0, xanchor='left', yanchor='top',
                 pad=dict(t=45), showactive=False,
                 buttons=[dict(label='▶', method='animate', args=[None, play]),
                          dict(label='⏸', method='animate', args=[[None], quieto]),
                          dict(label='⏮', method='animate', args=[['0'], quieto])]),
            dict(type='buttons', direction='left', x=0.5, y=1.0, xanchor='center', yanchor='bottom',
                 showactive=True,
                 buttons=[dict(label=k, method='relayout', args=[{'scene.camera': c}])
                          for k, c in CAMARAS.items()])],
        sliders=[dict(active=0, x=0.12, len=0.86, y=0.0, yanchor='top', pad=dict(t=30),
                      currentvalue=dict(visible=False),
                      steps=[dict(label=f'{tf[i]:.1f}' if i % 10 == 0 else '', method='animate', args=[[str(i)], quieto])
                             for i in range(len(tf))])])
    return fig

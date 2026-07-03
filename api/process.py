"""
=============================================================================
ELIMINACIÓN DE RUIDO EN AUDIO — v2.0
=============================================================================
Universidad Nacional de Colombia — Teoría de la Información 2026-1
Juan David Montenegro Lopez | Cristian David Arcia Quintero
=============================================================================

DESCRIPCIÓN:
  Este programa carga un archivo de audio con ruido de fondo y aplica
  técnicas de procesamiento digital de señales para eliminarlo.

ALGORITMO PRINCIPAL:
  MMSE no estacionario (noisereduce) — detecta automáticamente los 
  segmentos de silencio del audio para estimar el perfil del ruido,
  luego aplica una máscara espectral adaptativa frame a frame.

  También disponibles: sustracción espectral clásica y filtro Wiener.

USO:
  python eliminador_ruido_v2.py <audio.wav>
  python eliminador_ruido_v2.py <audio.wav> --metodo espectral
  python eliminador_ruido_v2.py <audio.wav> --agresividad 1.2
  python eliminador_ruido_v2.py --demo

DEPENDENCIAS:
  pip install numpy scipy matplotlib soundfile librosa noisereduce
=============================================================================
"""

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fft import fft, ifft, fftfreq
import argparse, os, sys, warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
#  CARGA / GUARDADO
# ─────────────────────────────────────────────────────────────────────────────

def cargar_audio(ruta):
    try:
        import librosa
        s, fs = librosa.load(ruta, sr=None, mono=True)
    except Exception:
        s, fs = sf.read(ruta)
        if s.ndim > 1:
            s = s.mean(axis=1)
    print(f"  ✔ {os.path.basename(ruta)}  |  {fs} Hz  |  {len(s)/fs:.2f}s")
    return s.astype(np.float32), fs

def guardar_audio(s, fs, ruta):
    s = s / (np.max(np.abs(s)) + 1e-9)
    sf.write(ruta, s, fs)
    print(f"  ✔ Guardado: {ruta}")


# ─────────────────────────────────────────────────────────────────────────────
#  ESTIMACIÓN AUTOMÁTICA DEL PERFIL DE RUIDO
# ─────────────────────────────────────────────────────────────────────────────

def estimar_ruido(señal, fs, percentil=15):
    """
    Detecta automáticamente los segmentos más silenciosos del audio
    (donde solo hay ruido de fondo) y los concatena para obtener
    un perfil espectral del ruido.
    """
    ventana = int(0.1 * fs)   # ventanas de 100 ms
    n = len(señal) // ventana
    rms = np.array([np.sqrt(np.mean(señal[i*ventana:(i+1)*ventana]**2))
                    for i in range(n)])
    umbral = np.percentile(rms, percentil)
    idx = [i for i, r in enumerate(rms) if r <= umbral]
    if not idx:
        idx = [np.argmin(rms)]
    ruido = np.concatenate([señal[i*ventana:(i+1)*ventana] for i in idx])
    print(f"  → Ruido estimado a partir de {len(idx)*0.1:.1f}s de segmentos silenciosos")
    return ruido


# ─────────────────────────────────────────────────────────────────────────────
#  MÉTODO 1 — MMSE NO ESTACIONARIO (principal, mejor calidad)
# ─────────────────────────────────────────────────────────────────────────────

def reducir_mmse(señal, fs, agresividad=1.0, **_):
    """
    Utiliza la librería noisereduce que implementa una sustracción
    espectral no estacionaria con estimación de SNR por bin.
    agresividad: 0.5 (suave) — 1.0 (estándar) — 1.5 (agresivo)
    """
    import noisereduce as nr
    ruido = estimar_ruido(señal, fs)
    resultado = nr.reduce_noise(
        y=señal, sr=fs,
        y_noise=ruido,
        prop_decrease=np.clip(agresividad, 0.1, 2.0),
        stationary=False,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        time_constant_s=0.5,
        freq_mask_smooth_hz=500,
    )
    return resultado


# ─────────────────────────────────────────────────────────────────────────────
#  MÉTODO 2 — SUSTRACCIÓN ESPECTRAL CLÁSICA
# ─────────────────────────────────────────────────────────────────────────────

def reducir_espectral(señal, fs, agresividad=2.0, **_):
    """
    Estima el espectro del ruido y lo resta del espectro de la señal.
    Incluye suavizado temporal para evitar artefactos musicales.
    alfa controla la agresividad de la sustracción.
    """
    TAM = 1024
    SALTO = TAM // 4
    vent = np.sqrt(np.hanning(TAM))

    ruido = estimar_ruido(señal, fs)
    n_r = (len(ruido) - TAM) // SALTO
    perfil = np.zeros(TAM)
    for i in range(max(n_r, 1)):
        f = ruido[i*SALTO:i*SALTO+TAM] * vent
        perfil += np.abs(fft(f))**2
    perfil /= max(n_r, 1)

    n_frames = (len(señal) - TAM) // SALTO + 1
    salida = np.zeros(len(señal) + TAM)
    peso   = np.zeros(len(señal) + TAM)
    G_prev = np.ones(TAM)

    for i in range(n_frames):
        ini = i * SALTO
        frame = señal[ini:ini+TAM] * vent
        X = fft(frame)
        mag2 = np.abs(X)**2
        fase = np.angle(X)

        mag2_limpia = np.maximum(mag2 - agresividad * perfil, 0.005 * perfil)
        G = np.sqrt(mag2_limpia / (mag2 + 1e-10))
        G = 0.9 * G_prev + 0.1 * G   # suavizado temporal
        G_prev = G.copy()

        frame_limpio = np.real(ifft(G * X)) * vent
        salida[ini:ini+TAM] += frame_limpio
        peso[ini:ini+TAM]   += vent**2

    m = peso > 1e-8
    salida[m] /= peso[m]
    return salida[:len(señal)]


# ─────────────────────────────────────────────────────────────────────────────
#  MÉTODO 3 — FILTRO DE WIENER
# ─────────────────────────────────────────────────────────────────────────────

def reducir_wiener(señal, fs, agresividad=1.5, **_):
    """
    Ganancia óptima de Wiener: H = SNR / (1 + SNR) por bin de frecuencia.
    """
    TAM = 1024
    SALTO = TAM // 4
    vent = np.sqrt(np.hanning(TAM))

    ruido = estimar_ruido(señal, fs)
    n_r = (len(ruido) - TAM) // SALTO
    perfil = np.zeros(TAM)
    for i in range(max(n_r, 1)):
        f = ruido[i*SALTO:i*SALTO+TAM] * vent
        perfil += np.abs(fft(f))**2
    perfil /= max(n_r, 1)

    n_frames = (len(señal) - TAM) // SALTO + 1
    salida = np.zeros(len(señal) + TAM)
    peso   = np.zeros(len(señal) + TAM)

    for i in range(n_frames):
        ini = i * SALTO
        frame = señal[ini:ini+TAM] * vent
        X = fft(frame)
        Px = np.abs(X)**2
        H = np.maximum(0, 1 - agresividad * perfil / (Px + 1e-10))
        frame_limpio = np.real(ifft(H * X)) * vent
        salida[ini:ini+TAM] += frame_limpio
        peso[ini:ini+TAM]   += vent**2

    m = peso > 1e-8
    salida[m] /= peso[m]
    return salida[:len(señal)]


# ─────────────────────────────────────────────────────────────────────────────
#  MEJORA DE CALIDAD
# ─────────────────────────────────────────────────────────────────────────────
def mejorar_calidad(señal, fs):
    from scipy.signal import sosfilt

    s = np.array(señal, dtype=np.float64, copy=True)

    def biquad_lowshelf(f0, ganancia_db, fs):
        A = 10**(ganancia_db/40); w0 = 2*np.pi*f0/fs
        alpha = np.sin(w0)/2 * np.sqrt((A+1/A)*(1/0.9-1)+2)
        b0=A*((A+1)-(A-1)*np.cos(w0)+2*np.sqrt(A)*alpha); b1=2*A*((A-1)-(A+1)*np.cos(w0)); b2=A*((A+1)-(A-1)*np.cos(w0)-2*np.sqrt(A)*alpha)
        a0=(A+1)+(A-1)*np.cos(w0)+2*np.sqrt(A)*alpha; a1=-2*((A-1)+(A+1)*np.cos(w0)); a2=(A+1)+(A-1)*np.cos(w0)-2*np.sqrt(A)*alpha
        return np.array([[b0/a0,b1/a0,b2/a0,1,a1/a0,a2/a0]])

    def biquad_peak(f0, ganancia_db, Q, fs):
        A=10**(ganancia_db/40); w0=2*np.pi*f0/fs; alpha=np.sin(w0)/(2*Q)
        b0=1+alpha*A; b1=-2*np.cos(w0); b2=1-alpha*A
        a0=1+alpha/A; a1=-2*np.cos(w0); a2=1-alpha/A
        return np.array([[b0/a0,b1/a0,b2/a0,1,a1/a0,a2/a0]])

    def biquad_highshelf(f0, ganancia_db, fs):
        A=10**(ganancia_db/40); w0=2*np.pi*f0/fs
        alpha=np.sin(w0)/2*np.sqrt((A+1/A)*(1/0.9-1)+2)
        b0=A*((A+1)+(A-1)*np.cos(w0)+2*np.sqrt(A)*alpha); b1=-2*A*((A-1)+(A+1)*np.cos(w0)); b2=A*((A+1)+(A-1)*np.cos(w0)-2*np.sqrt(A)*alpha)
        a0=(A+1)-(A-1)*np.cos(w0)+2*np.sqrt(A)*alpha; a1=2*((A-1)-(A+1)*np.cos(w0)); a2=(A+1)-(A-1)*np.cos(w0)-2*np.sqrt(A)*alpha
        return np.array([[b0/a0,b1/a0,b2/a0,1,a1/a0,a2/a0]])

    s = sosfilt(biquad_lowshelf(300, 5.0, fs), s)
    s = sosfilt(biquad_peak(1500, 4.0, 0.9, fs), s)
    s = sosfilt(biquad_highshelf(3000, 4.0, fs), s)
    rms_orig = np.sqrt(np.mean(señal**2)) + 1e-9
    rms_proc = np.sqrt(np.mean(s**2)) + 1e-9
    resultado = s * float(rms_orig/rms_proc)
    return np.array(resultado, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def calcular_snr_relativa(señal_orig, señal_proc, fs):
    """Calcula la reducción de ruido en los segmentos silenciosos."""
    ventana = int(0.1 * fs)
    n = len(señal_orig) // ventana
    rms = np.array([np.sqrt(np.mean(señal_orig[i*ventana:(i+1)*ventana]**2)) for i in range(n)])
    umbral = np.percentile(rms, 15)
    idx = [i for i, r in enumerate(rms) if r <= umbral]
    if not idx:
        return 0.0
    ruido_orig = np.concatenate([señal_orig[i*ventana:(i+1)*ventana] for i in idx])
    ruido_proc = np.concatenate([señal_proc[i*ventana:(i+1)*ventana] for i in idx if (i+1)*ventana <= len(señal_proc)])
    if len(ruido_proc) == 0:
        return 0.0
    r1 = np.sqrt(np.mean(ruido_orig**2))
    r2 = np.sqrt(np.mean(ruido_proc**2))
    return 20 * np.log10(r1 / (r2 + 1e-10))


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def graficar(señal_orig, señal_proc, fs, metodo, reduccion_db, ruta_png):
    FONDO = '#161b22'; GRID = '#30363d'; TEXTO = '#e6edf3'
    ORIG = '#ff6b6b'; PROC = '#4ecdc4'; RUIDO = '#ffd93d'

    fig = plt.figure(figsize=(16, 10), facecolor='#0d1117')
    t = np.linspace(0, len(señal_orig)/fs, len(señal_orig))

    # Señales en el tiempo
    for col, (s, color, label) in enumerate([
        (señal_orig, ORIG, 'Original (con ruido)'),
        (señal_proc, PROC, f'Procesada — {metodo}')
    ]):
        ax = fig.add_subplot(3, 2, col+1, facecolor=FONDO)
        ax.plot(t, s, color=color, lw=0.25, alpha=0.85)
        ax.set_title(label, color=TEXTO, fontsize=10)
        ax.set_xlabel('Tiempo (s)', color=TEXTO, fontsize=8)
        ax.set_ylabel('Amplitud', color=TEXTO, fontsize=8)
        ax.tick_params(colors=TEXTO, labelsize=7)
        ax.grid(True, color=GRID, alpha=0.5)
        for sp in ax.spines.values(): sp.set_color(GRID)

    # Espectros FFT
    for col, (s, color, label) in enumerate([
        (señal_orig, ORIG, 'Espectro — Original'),
        (señal_proc, PROC, 'Espectro — Procesada')
    ]):
        ax = fig.add_subplot(3, 2, col+3, facecolor=FONDO)
        N = len(s); E = np.abs(fft(s))[:N//2]; F = fftfreq(N, 1/fs)[:N//2]
        E_db = 20*np.log10(E + 1e-10)
        ax.fill_between(F, E_db, alpha=0.25, color=color)
        ax.plot(F, E_db, color=color, lw=0.5)
        ax.set_title(label, color=TEXTO, fontsize=10)
        ax.set_xlabel('Frecuencia (Hz)', color=TEXTO, fontsize=8)
        ax.set_ylabel('Magnitud (dB)', color=TEXTO, fontsize=8)
        ax.set_xlim([0, 4000])
        ax.tick_params(colors=TEXTO, labelsize=7)
        ax.grid(True, color=GRID, alpha=0.5)
        for sp in ax.spines.values(): sp.set_color(GRID)

    # PSD comparativa
    ax5 = fig.add_subplot(3, 1, 3, facecolor=FONDO)
    ventana = int(0.1 * fs)
    n = len(señal_orig) // ventana
    rms = np.array([np.sqrt(np.mean(señal_orig[i*ventana:(i+1)*ventana]**2)) for i in range(n)])
    idx_r = [i for i, r in enumerate(rms) if r <= np.percentile(rms, 15)]
    ruido = np.concatenate([señal_orig[i*ventana:(i+1)*ventana] for i in idx_r])

    for s, color, label in [
        (señal_orig, ORIG, 'Original'),
        (señal_proc, PROC, f'Procesada ({metodo})'),
        (ruido, RUIDO, 'Ruido estimado'),
    ]:
        fr, p = welch(s, fs, nperseg=4096)
        ax5.semilogy(fr, p, color=color, lw=0.9, label=label, alpha=0.9)

    ax5.set_title('PSD Comparativa — Original / Procesada / Ruido', color=TEXTO, fontsize=10)
    ax5.set_xlabel('Frecuencia (Hz)', color=TEXTO, fontsize=8)
    ax5.set_xlim([0, 4000])
    ax5.tick_params(colors=TEXTO, labelsize=7)
    ax5.grid(True, color=GRID, alpha=0.5)
    ax5.legend(facecolor=FONDO, edgecolor=GRID, labelcolor=TEXTO, fontsize=9)
    for sp in ax5.spines.values(): sp.set_color(GRID)

    fig.suptitle(
        f'Eliminación de Ruido en Audio — {metodo}\n'
        f'Reducción de ruido en segmentos silenciosos: {reduccion_db:.1f} dB',
        color=TEXTO, fontsize=13, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(ruta_png, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  ✔ Gráfica: {ruta_png}")


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

METODOS = {
    'mmse':      (reducir_mmse,      'MMSE No Estacionario'),
    'espectral': (reducir_espectral, 'Sustracción Espectral'),
    'wiener':    (reducir_wiener,    'Filtro de Wiener'),
}

SEP = '─' * 62

def procesar(ruta_entrada, metodo='mmse', agresividad=1.0,
             ruta_salida=None, graficar_flag=True):

    print(f'\n{SEP}')
    print('  ELIMINADOR DE RUIDO — Teoría de la Información UNAL 2026-1')
    print(SEP)

    señal, fs = cargar_audio(ruta_entrada)
    base = os.path.splitext(ruta_entrada)[0]

    mets = list(METODOS.keys()) if metodo == 'todos' else [metodo]

    for m in mets:
        if m not in METODOS:
            print(f'  ✘ Método desconocido: {m}'); continue

        fn, nombre = METODOS[m]
        print(f'\n[→] Método: {nombre}  |  agresividad={agresividad}')

        try:
            resultado = fn(señal, fs, agresividad=agresividad)
        except Exception as e:
            print(f'  ✘ Error: {e}'); import traceback; traceback.print_exc(); continue

        reduccion = calcular_snr_relativa(señal, resultado, fs)
        print(f'  → Reducción de ruido medida: {reduccion:.1f} dB')

        sal = ruta_salida or f'{base}_sin_ruido_{m}.wav'
        guardar_audio(resultado, fs, sal)

        if graficar_flag:
            png = f'{base}_analisis_{m}.png'
            graficar(señal, resultado, fs, nombre, reduccion, png)

    print(f'\n{SEP}')
    print('  ✔ Listo.')
    print(SEP + '\n')


# ─────────────────────────────────────────────────────────────────────────────
#  AUDIO DE PRUEBA
# ─────────────────────────────────────────────────────────────────────────────

def generar_demo(ruta='audio_demo.wav'):
    fs = 44100; dur = 6.0
    t = np.linspace(0, dur, int(fs*dur))
    # Voz sintética (formantes aproximados)
    voz = (0.5*np.sin(2*np.pi*200*t) + 0.4*np.sin(2*np.pi*400*t) +
           0.3*np.sin(2*np.pi*800*t) + 0.15*np.sin(2*np.pi*1200*t))
    # Envolvente tipo palabra (silencio - habla - silencio - habla)
    env = np.ones(len(t))
    env[:int(0.5*fs)] = 0.05      # silencio inicial
    env[int(2*fs):int(2.5*fs)] = 0.05   # pausa
    env[int(4.5*fs):] = 0.05     # silencio final
    voz *= env
    # Ruido: hum 60Hz + ruido de sala broadband
    ruido = (0.15*np.sin(2*np.pi*60*t) + 0.08*np.sin(2*np.pi*120*t) +
             0.12*np.random.randn(len(t)))
    señal = voz + ruido
    señal /= np.max(np.abs(señal)) + 1e-9
    sf.write(ruta, señal, fs)
    print(f'  ✔ Audio demo generado: {ruta}')
    return ruta


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Eliminador de ruido en audio — UNAL Teoría de la Información',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python eliminador_ruido_v2.py grabacion.wav
  python eliminador_ruido_v2.py grabacion.wav --metodo espectral
  python eliminador_ruido_v2.py grabacion.wav --metodo todos
  python eliminador_ruido_v2.py grabacion.wav --agresividad 1.3
  python eliminador_ruido_v2.py --demo
        """
    )
    p.add_argument('audio', nargs='?', help='Archivo de audio de entrada')
    p.add_argument('--metodo', default='mmse',
                   choices=['mmse','espectral','wiener','todos'],
                   help='Algoritmo de reducción (default: mmse)')
    p.add_argument('--agresividad', type=float, default=1.0,
                   help='Intensidad del filtrado 0.5=suave 1.0=normal 1.5=agresivo (default: 1.0)')
    p.add_argument('--salida', default=None, help='Ruta del archivo de salida')
    p.add_argument('--sin-graficas', action='store_true')
    p.add_argument('--demo', action='store_true', help='Generar y procesar audio de prueba')

    args = p.parse_args()

    if args.demo:
        ruta = generar_demo()
        procesar(ruta, metodo='todos', agresividad=1.0,
                 graficar_flag=not args.sin_graficas)
    elif args.audio:
        procesar(args.audio, metodo=args.metodo,
                 agresividad=args.agresividad,
                 ruta_salida=args.salida,
                 graficar_flag=not args.sin_graficas)
    else:
        p.print_help()
        print('\n  Tip: usa --demo para probar sin tener un archivo.')


# ─────────────────────────────────────────────────────────────────────────────
#  HANDLER VERCEL — solo se activa cuando corre como función serverless
# ─────────────────────────────────────────────────────────────────────────────
from http.server import BaseHTTPRequestHandler
import io, cgi, json

class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        try:
            ctype, pdict = cgi.parse_header(self.headers.get('Content-Type',''))
            pdict['boundary'] = bytes(pdict['boundary'], 'utf-8')
            fields = cgi.parse_multipart(self.rfile, pdict)

            audio_bytes = fields.get('audio', [None])[0]
            if audio_bytes is None:
                self._error(400, 'No se recibió el campo "audio"'); return

            metodo      = fields.get('metodo', [b'espectral'])[0]
            agresividad = fields.get('agresividad', [b'1.5'])[0]
            if isinstance(metodo, bytes):      metodo = metodo.decode()
            if isinstance(agresividad, bytes): agresividad = float(agresividad.decode())
            if metodo not in METODOS:          metodo = 'espectral'

            buf_in = io.BytesIO(audio_bytes if isinstance(audio_bytes, bytes) else bytes(audio_bytes))
            senal, fs = sf.read(buf_in)
            if senal.ndim > 1: senal = senal.mean(axis=1)
            senal = senal.astype(np.float32)
            senal = np.ascontiguousarray(senal, dtype=np.float64)

            # Limitar a 60s (timeout Vercel free)
            if len(senal) > 60 * fs:
                senal = senal[:int(60*fs)]

            fn, _ = METODOS[metodo]
            resultado = fn(senal, fs, agresividad=agresividad)
            resultado = mejorar_calidad(resultado, fs)
            snr = calcular_snr_relativa(senal, resultado, fs)

            resultado = resultado / (np.max(np.abs(resultado)) + 1e-9)
            buf_out = io.BytesIO()
            sf.write(buf_out, resultado, fs, format='WAV', subtype='PCM_16')
            wav_bytes = buf_out.getvalue()

            self.send_response(200)
            self._cors()
            self.send_header('Content-Type', 'audio/wav')
            self.send_header('Content-Disposition', 'attachment; filename="audio_sin_ruido.wav"')
            self.send_header('Content-Length', str(len(wav_bytes)))
            self.send_header('X-SNR-Reduction', f'{snr:.1f}')
            self.send_header('X-Metodo', metodo)
            self.end_headers()
            self.wfile.write(wav_bytes)

        except Exception as e:
            self._error(500, str(e))

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Expose-Headers', 'X-SNR-Reduction, X-Metodo')

    def _error(self, code, msg):
        body = json.dumps({'error': msg}).encode()
        self.send_response(code)
        self._cors()
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    main()
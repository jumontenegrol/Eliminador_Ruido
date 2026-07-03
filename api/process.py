"""
API serverless — Eliminación de Ruido en Audio
Universidad Nacional de Colombia — Teoría de la Información 2026-1

Recibe un archivo WAV via POST multipart/form-data
Devuelve el audio procesado como WAV descargable.
"""

from http.server import BaseHTTPRequestHandler
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.fft import fft, ifft
from scipy.signal import sosfilt, butter
import io, cgi, json


# ─── ESTIMACIÓN DE RUIDO ────────────────────────────────────────────────────

def estimar_ruido(senal, fs, percentil=15):
    ventana = int(0.1 * fs)
    n = len(senal) // ventana
    if n == 0:
        return senal
    rms = np.array([np.sqrt(np.mean(senal[i*ventana:(i+1)*ventana]**2))
                    for i in range(n)])
    umbral = np.percentile(rms, percentil)
    idx = [i for i, r in enumerate(rms) if r <= umbral]
    if not idx:
        idx = [int(np.argmin(rms))]
    return np.concatenate([senal[i*ventana:(i+1)*ventana] for i in idx])


# ─── SUSTRACCIÓN ESPECTRAL ──────────────────────────────────────────────────

def reducir_espectral(senal, fs, agresividad=2.0):
    TAM = 1024
    SALTO = TAM // 4
    vent = np.sqrt(np.hanning(TAM))

    ruido = estimar_ruido(senal, fs)
    n_r = max((len(ruido) - TAM) // SALTO, 1)
    perfil = np.zeros(TAM)
    for i in range(n_r):
        f = ruido[i*SALTO:i*SALTO+TAM]
        if len(f) < TAM:
            f = np.pad(f, (0, TAM-len(f)))
        perfil += np.abs(fft(f * vent))**2
    perfil /= n_r

    n_frames = (len(senal) - TAM) // SALTO + 1
    salida = np.zeros(len(senal) + TAM)
    peso   = np.zeros(len(senal) + TAM)
    G_prev = np.ones(TAM)

    for i in range(n_frames):
        ini = i * SALTO
        frame = senal[ini:ini+TAM]
        if len(frame) < TAM:
            frame = np.pad(frame, (0, TAM-len(frame)))
        frame = frame * vent
        X = fft(frame)
        mag2 = np.abs(X)**2
        mag2_limpia = np.maximum(mag2 - agresividad*perfil, 0.005*perfil)
        G = np.sqrt(mag2_limpia / (mag2 + 1e-10))
        G = 0.9*G_prev + 0.1*G
        G_prev = G.copy()
        frame_limpio = np.real(ifft(G * X)) * vent
        salida[ini:ini+TAM] += frame_limpio
        peso[ini:ini+TAM]   += vent**2

    m = peso > 1e-8
    salida[m] /= peso[m]
    return salida[:len(senal)]


# ─── FILTRO DE WIENER ───────────────────────────────────────────────────────

def reducir_wiener(senal, fs, agresividad=1.5):
    TAM = 1024
    SALTO = TAM // 4
    vent = np.sqrt(np.hanning(TAM))

    ruido = estimar_ruido(senal, fs)
    n_r = max((len(ruido) - TAM) // SALTO, 1)
    perfil = np.zeros(TAM)
    for i in range(n_r):
        f = ruido[i*SALTO:i*SALTO+TAM]
        if len(f) < TAM:
            f = np.pad(f, (0, TAM-len(f)))
        perfil += np.abs(fft(f * vent))**2
    perfil /= n_r

    n_frames = (len(senal) - TAM) // SALTO + 1
    salida = np.zeros(len(senal) + TAM)
    peso   = np.zeros(len(senal) + TAM)

    for i in range(n_frames):
        ini = i * SALTO
        frame = senal[ini:ini+TAM]
        if len(frame) < TAM:
            frame = np.pad(frame, (0, TAM-len(frame)))
        frame = frame * vent
        X = fft(frame)
        Px = np.abs(X)**2
        H = np.maximum(0, 1 - agresividad*perfil/(Px + 1e-10))
        frame_limpio = np.real(ifft(H * X)) * vent
        salida[ini:ini+TAM] += frame_limpio
        peso[ini:ini+TAM]   += vent**2

    m = peso > 1e-8
    salida[m] /= peso[m]
    return salida[:len(senal)]


# ─── MMSE ───────────────────────────────────────────────────────────────────

def reducir_mmse(senal, fs, agresividad=1.0):
    ruido = estimar_ruido(senal, fs)
    return nr.reduce_noise(
        y=senal, sr=fs, y_noise=ruido,
        prop_decrease=float(np.clip(agresividad, 0.1, 2.0)),
        stationary=False, n_fft=1024, hop_length=256,
        time_constant_s=0.5, freq_mask_smooth_hz=500,
    )


# ─── EQ CORRECTIVA ──────────────────────────────────────────────────────────

def mejorar_calidad(senal, fs):
    s = senal.copy().astype(np.float64)

    def biquad_lowshelf(f0, ganancia_db, fs):
        A = 10**(ganancia_db/40)
        w0 = 2*np.pi*f0/fs
        alpha = np.sin(w0)/2 * np.sqrt((A+1/A)*(1/0.9-1)+2)
        b0 = A*((A+1)-(A-1)*np.cos(w0)+2*np.sqrt(A)*alpha)
        b1 = 2*A*((A-1)-(A+1)*np.cos(w0))
        b2 = A*((A+1)-(A-1)*np.cos(w0)-2*np.sqrt(A)*alpha)
        a0 = (A+1)+(A-1)*np.cos(w0)+2*np.sqrt(A)*alpha
        a1 = -2*((A-1)+(A+1)*np.cos(w0))
        a2 = (A+1)+(A-1)*np.cos(w0)-2*np.sqrt(A)*alpha
        return np.array([[b0/a0,b1/a0,b2/a0,1,a1/a0,a2/a0]])

    def biquad_peak(f0, ganancia_db, Q, fs):
        A = 10**(ganancia_db/40)
        w0 = 2*np.pi*f0/fs
        alpha = np.sin(w0)/(2*Q)
        b0 = 1+alpha*A; b1 = -2*np.cos(w0); b2 = 1-alpha*A
        a0 = 1+alpha/A; a1 = -2*np.cos(w0); a2 = 1-alpha/A
        return np.array([[b0/a0,b1/a0,b2/a0,1,a1/a0,a2/a0]])

    def biquad_highshelf(f0, ganancia_db, fs):
        A = 10**(ganancia_db/40)
        w0 = 2*np.pi*f0/fs
        alpha = np.sin(w0)/2 * np.sqrt((A+1/A)*(1/0.9-1)+2)
        b0 = A*((A+1)+(A-1)*np.cos(w0)+2*np.sqrt(A)*alpha)
        b1 = -2*A*((A-1)+(A+1)*np.cos(w0))
        b2 = A*((A+1)+(A-1)*np.cos(w0)-2*np.sqrt(A)*alpha)
        a0 = (A+1)-(A-1)*np.cos(w0)+2*np.sqrt(A)*alpha
        a1 = 2*((A-1)-(A+1)*np.cos(w0))
        a2 = (A+1)-(A-1)*np.cos(w0)-2*np.sqrt(A)*alpha
        return np.array([[b0/a0,b1/a0,b2/a0,1,a1/a0,a2/a0]])

    s = sosfilt(biquad_lowshelf(300, 5.0, fs), s)
    s = sosfilt(biquad_peak(1500, 4.0, 0.9, fs), s)
    s = sosfilt(biquad_highshelf(3000, 4.0, fs), s)

    rms_orig = np.sqrt(np.mean(senal**2)) + 1e-9
    rms_proc = np.sqrt(np.mean(s**2)) + 1e-9
    return (s * (rms_orig/rms_proc)).astype(np.float32)


# ─── CALCULAR SNR ───────────────────────────────────────────────────────────

def calcular_snr(senal_orig, senal_proc, fs):
    ventana = int(0.1 * fs)
    n = len(senal_orig) // ventana
    if n == 0:
        return 0.0
    rms = np.array([np.sqrt(np.mean(senal_orig[i*ventana:(i+1)*ventana]**2)) for i in range(n)])
    umbral = np.percentile(rms, 15)
    idx = [i for i, r in enumerate(rms) if r <= umbral]
    if not idx:
        return 0.0
    ro = np.concatenate([senal_orig[i*ventana:(i+1)*ventana] for i in idx])
    rp = np.concatenate([senal_proc[i*ventana:(i+1)*ventana] for i in idx if (i+1)*ventana <= len(senal_proc)])
    if len(rp) == 0:
        return 0.0
    return float(20 * np.log10(np.sqrt(np.mean(ro**2)) / (np.sqrt(np.mean(rp**2)) + 1e-10)))


# ─── HANDLER VERCEL ─────────────────────────────────────────────────────────

METODOS = {
    'espectral': reducir_espectral,
    'wiener':    reducir_wiener,
    'mmse':      reducir_mmse,
}

class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        try:
            ctype, pdict = cgi.parse_header(self.headers.get('Content-Type', ''))
            if ctype != 'multipart/form-data':
                self._error(400, 'Se esperaba multipart/form-data')
                return

            pdict['boundary'] = bytes(pdict['boundary'], 'utf-8')
            fields = cgi.parse_multipart(self.rfile, pdict)

            # Leer archivo de audio
            audio_bytes = fields.get('audio', [None])[0]
            if audio_bytes is None:
                self._error(400, 'No se recibió el campo "audio"')
                return

            metodo     = fields.get('metodo', [b'espectral'])[0]
            agresividad= fields.get('agresividad', [b'1.5'])[0]
            if isinstance(metodo, bytes):
                metodo = metodo.decode()
            if isinstance(agresividad, bytes):
                agresividad = float(agresividad.decode())

            if metodo not in METODOS:
                metodo = 'espectral'

            # Decodificar audio
            buf_in = io.BytesIO(audio_bytes if isinstance(audio_bytes, bytes) else bytes(audio_bytes))
            senal, fs = sf.read(buf_in)
            if senal.ndim > 1:
                senal = senal.mean(axis=1)
            senal = senal.astype(np.float32)

            # Limitar a 60 segundos (límite de función serverless)
            max_muestras = 60 * fs
            if len(senal) > max_muestras:
                senal = senal[:int(max_muestras)]

            # Procesar
            fn = METODOS[metodo]
            resultado = fn(senal, fs, agresividad=agresividad)
            resultado = mejorar_calidad(resultado, fs)

            # Calcular SNR
            snr = calcular_snr(senal, resultado, fs)

            # Normalizar y exportar
            resultado = resultado / (np.max(np.abs(resultado)) + 1e-9)
            buf_out = io.BytesIO()
            sf.write(buf_out, resultado, fs, format='WAV', subtype='PCM_16')
            wav_bytes = buf_out.getvalue()

            # Responder
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

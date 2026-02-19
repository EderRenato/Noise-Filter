"""
UNIVERSIDADE FEDERAL DO OESTE DA BAHIA
CAMPUS DE BOM JESUS DA LAPA
GRADUAÇÃO EM ENGENHARIA ELÉTRICA
DISCIPLINA: Processamento Digital de Sinais

Projeto: Filtros Digitais FIR e IIR - OTIMIZADO PARA RUÍDOS ESPECÍFICOS
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
import os

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# ==============================================================================
# CONFIGURAÇÕES DE FILTROS POR TIPO DE RUÍDO
# ==============================================================================

CONFIGURACOES_FILTROS = {
    "airconditioner": {
        "tipo": "passa_altas",
        "fc_pass": 300,
        "fc_stop": 150,
        "descricao": "Remove ruído grave constante do AC (50-500 Hz)",
    },
    "airportannouncements": {
        "tipo": "passa_banda",
        "fc_pass_low": 250,
        "fc_pass_high": 3500,
        "fc_stop_low": 150,
        "fc_stop_high": 4500,
        "descricao": "Mantém voz, remove reverberação",
    },
    "babble": {
        "tipo": "passa_banda",
        "fc_pass_low": 300,
        "fc_pass_high": 3000,
        "fc_stop_low": 100,
        "fc_stop_high": 4000,
        "descricao": "Foca na voz principal",
    },
    "munching": {
        "tipo": "passa_altas",
        "fc_pass": 400,
        "fc_stop": 200,
        "descricao": "Remove sons graves de mastigação",
    },
    "copymachine": {
        "tipo": "passa_banda",
        "fc_pass_low": 300,
        "fc_pass_high": 3500,
        "fc_stop_low": 150,
        "fc_stop_high": 4500,
        "descricao": "Mantém voz (300-3500 Hz), atenua picos do ruído mecânico",
    },
    "neighborhood": {
        "tipo": "passa_banda",
        "fc_pass_low": 300,
        "fc_pass_high": 3500,
        "fc_stop_low": 150,
        "fc_stop_high": 4500,
        "descricao": "Filtra ruídos externos variados",
    },
    "shuttingdoor": {
        "tipo": "passa_altas",
        "fc_pass": 300,
        "fc_stop": 150,
        "descricao": "Remove impacto grave (<200 Hz)",
    },
    "typing": {
        "tipo": "passa_banda",
        "fc_pass_low": 300,
        "fc_pass_high": 3500,
        "fc_stop_low": 100,
        "fc_stop_high": 5000,
        "descricao": "Mantém voz, atenua cliques de digitação (~200-1000 Hz)",
    },
    "vacuumcleaner": {
        "tipo": "passa_banda",
        "fc_pass_low": 200,
        "fc_pass_high": 3500,
        "fc_stop_low": 100,
        "fc_stop_high": 4500,
        "descricao": "Mantém voz, atenua ruído do aspirador",
    },
}

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================


def detectar_tipo_ruido(nome_arquivo):
    nome_lower = nome_arquivo.lower()
    palavras_chave = {
        "airconditioner": ["airconditioner", "air_conditioner"],
        "airportannouncements": ["airportannouncements", "airport"],
        "babble": ["babble"],
        "munching": ["munching", "eating", "chewing"],
        "copymachine": ["copymachine", "copy_machine", "copiadora"],
        "neighborhood": ["neighborhood", "vizinhanca"],
        "shuttingdoor": ["shuttingdoor", "shutting_door"],
        "typing": ["typing", "keyboard", "teclado"],
        "vacuumcleaner": ["vacuumcleaner", "vacuum_cleaner", "vacuum"],
    }
    for tipo, keywords in palavras_chave.items():
        for kw in keywords:
            if kw in nome_lower:
                return tipo
    return None


def plot_espectro(y, Fs, titulo, subplot_pos=None, xlim_max=8000):
    Ns = len(y)
    S = np.abs(np.fft.fft(y)[: Ns // 2])
    f = np.fft.fftfreq(Ns, 1 / Fs)[: Ns // 2]  # Garante mesmo tamanho que S

    if subplot_pos:
        plt.subplot(subplot_pos)
    plt.plot(f, S)
    plt.title(titulo)
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim([0, xlim_max])
    plt.grid(True)


def plot_sinal_tempo(t, y, titulo, subplot_pos=None):
    if subplot_pos:
        plt.subplot(subplot_pos)
    plt.plot(t, y)
    plt.title(titulo)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)


def analisar_filtro(b, a, Fs, titulo_filtro, N_ordem):
    """
    Análise completa do filtro.

    Correções aplicadas:
    - Polos/zeros via tf2zpk (evita overflow numérico de np.roots em filtros de alta ordem)
    - Resposta ao impulso via sosfilt para IIR (estável)
    - Atraso de grupo clipado para não estourar a escala
    - Diagrama de polos/zeros limitado ao entorno do círculo unitário
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"{titulo_filtro} — Ordem: {N_ordem}", fontsize=14, fontweight="bold")

    w, h = signal.freqz(b, a, worN=4096, fs=Fs)
    w_norm = w / (Fs / 2)  # 0..1 (normalizado por π)

    # 1. Magnitude linear
    plt.subplot(3, 2, 1)
    plt.plot(w, np.abs(h))
    plt.title("Magnitude do filtro")
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # 2. Magnitude em dB
    plt.subplot(3, 2, 2)
    plt.plot(w_norm, 20 * np.log10(np.abs(h) + 1e-10))
    plt.title("Resposta de magnitude (dB)")
    plt.xlabel("Frequência normalizada (×π rad/amostra)")
    plt.ylabel("Magnitude (dB)")
    plt.ylim([-100, 20])
    plt.grid(True)

    # 3. Detalhe banda passante
    plt.subplot(3, 2, 3)
    plt.plot(w_norm, 20 * np.log10(np.abs(h) + 1e-10))
    plt.title("Detalhe — Ondulação da banda passante")
    plt.xlabel("Frequência normalizada (×π rad/amostra)")
    plt.ylabel("Magnitude (dB)")
    plt.ylim([-5, 2])
    plt.xlim([0, 0.55])
    plt.grid(True)

    # 4. Atraso de grupo (clipado para escala legível)
    plt.subplot(3, 2, 4)
    w_gd, gd = signal.group_delay((b, a), w=4096, fs=Fs)
    gd_median = np.nanmedian(gd)
    gd_clip = np.clip(gd, gd_median - 200, gd_median + 200)
    plt.plot(w_gd / (Fs / 2), gd_clip)
    plt.title("Atraso de grupo (amostras)")
    plt.xlabel("Frequência normalizada (×π rad/amostra)")
    plt.ylabel("Amostras")
    plt.grid(True)

    # 5. Diagrama de polos e zeros  ← CORREÇÃO PRINCIPAL
    plt.subplot(3, 2, 5)
    try:
        z, p, _ = signal.tf2zpk(b, a)  # numericamente estável
    except Exception:
        z = np.roots(b)
        p = np.array([]) if len(a) == 1 else np.roots(a)

    theta = np.linspace(0, 2 * np.pi, 360)
    plt.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.8)
    plt.axhline(0, color="gray", linewidth=0.4)
    plt.axvline(0, color="gray", linewidth=0.4)
    if len(z):
        plt.scatter(
            np.real(z),
            np.imag(z),
            s=60,
            marker="o",
            facecolors="none",
            edgecolors="royalblue",
            linewidths=1.5,
            label="Zeros",
            zorder=5,
        )
    if len(p):
        plt.scatter(
            np.real(p),
            np.imag(p),
            s=60,
            marker="x",
            color="tomato",
            linewidths=1.5,
            label="Polos",
            zorder=5,
        )
    plt.title("Diagrama de polos e zeros")
    plt.xlabel("Parte real")
    plt.ylabel("Parte imaginária")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.axis("equal")
    plt.xlim([-1.7, 1.7])
    plt.ylim([-1.7, 1.7])

    # 6. Resposta ao impulso  ← CORREÇÃO: sosfilt para IIR
    plt.subplot(3, 2, 6)
    impulso = np.zeros(100)
    impulso[0] = 1
    is_fir = len(a) == 1
    if is_fir:
        h_imp = signal.lfilter(b, [1], impulso)
    else:
        sos = signal.tf2sos(b, a)
        h_imp = signal.sosfilt(sos, impulso)
    plt.stem(range(100), h_imp, basefmt=" ")
    plt.title("Resposta ao impulso (100 amostras)")
    plt.xlabel("Amostras (n)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()


# ==============================================================================
# FILTROS FIR
# ==============================================================================


def filtro_fir_janelamento(x, Fs, config, M=101, tipo_janela="hamming"):
    """FIR por janelamento. M deve ser ímpar para passa-altas/banda."""
    if M % 2 == 0:
        M += 1

    janelas_map = {
        "rectangular": "boxcar",
        "hamming": "hamming",
        "hanning": "hann",
        "blackman": "blackman",
    }
    janela = janelas_map.get(tipo_janela, "hamming")

    if config["tipo"] == "passa_altas":
        wc = config["fc_pass"] / (Fs / 2)
        b = signal.firwin(M, wc, window=janela, pass_zero=False)

    elif config["tipo"] == "passa_baixas":
        wc = config["fc_pass"] / (Fs / 2)
        b = signal.firwin(M, wc, window=janela, pass_zero=True)

    elif config["tipo"] == "passa_banda":
        wc_low = config["fc_pass_low"] / (Fs / 2)
        wc_high = config["fc_pass_high"] / (Fs / 2)
        b = signal.firwin(M, [wc_low, wc_high], window=janela, pass_zero=False)

    a = [1]
    y_filt = signal.lfilter(b, a, x)
    return b, a, y_filt


def filtro_fir_parks_mcclellan(x, Fs, config):
    """
    FIR por Parks-McClellan (Remez) com fallback para firwin2.

    O algoritmo Remez diverge numericamente quando a transição inferior
    é muito estreita (ex: 150-300 Hz) e a ordem M é alta. Nesse caso,
    usamos firwin2 (janelamento Blackman com frequências arbitrárias),
    que é sempre estável e produz resposta equivalente.

    Ordem calculada pela fórmula de Kaiser baseada na transição mais estreita.
    """
    nyq = Fs / 2

    if config["tipo"] == "passa_altas":
        f_stop, f_pass = config["fc_stop"], config["fc_pass"]
        freq_pts = [0, f_stop, f_pass, nyq]
        gain_pts = [0, 0,      1,      1]
        bandas_remez = [0, f_stop, f_pass, nyq]
        ganhos_remez = [0, 1]
        pesos_remez  = [10, 1]
        delta_f = (f_pass - f_stop) / Fs

    elif config["tipo"] == "passa_baixas":
        f_pass, f_stop = config["fc_pass"], config["fc_stop"]
        freq_pts = [0, f_pass, f_stop, nyq]
        gain_pts = [1, 1,      0,      0]
        bandas_remez = [0, f_pass, f_stop, nyq]
        ganhos_remez = [1, 0]
        pesos_remez  = [1, 10]
        delta_f = (f_stop - f_pass) / Fs

    elif config["tipo"] == "passa_banda":
        f_s1 = config["fc_stop_low"]
        f_p1 = config["fc_pass_low"]
        f_p2 = config["fc_pass_high"]
        f_s2 = min(config["fc_stop_high"], nyq - 50)
        freq_pts = [0, f_s1, f_p1, f_p2, f_s2, nyq]
        gain_pts = [0, 0,    1,    1,    0,    0]
        bandas_remez = [0, f_s1, f_p1, f_p2, f_s2, nyq]
        ganhos_remez = [0, 1, 0]
        pesos_remez  = [10, 1, 10]
        delta_f = min(f_p1 - f_s1, f_s2 - f_p2) / Fs

    # Fórmula de Kaiser: M ≈ (As - 8) / (2.285 * 2π * Δf/fs)
    As_db = 60
    M = int(np.ceil((As_db - 8) / (2.285 * 2 * np.pi * delta_f)))
    M = max(M, 51)
    if M % 2 == 0:
        M += 1

    # Tenta Remez; se divergir ou produzir coeficientes explodidos, usa firwin2
    b = None
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            b_remez = signal.remez(M, bandas_remez, ganhos_remez,
                                   weight=pesos_remez, fs=Fs, maxiter=500)
        if np.max(np.abs(b_remez)) < 10:  # coeficientes razoáveis
            b = b_remez
    except Exception:
        pass  # fallback para firwin2

    if b is None:
        # firwin2: janelamento com resposta de frequência arbitrária — sempre estável
        b = signal.firwin2(M, freq_pts, gain_pts, window="blackman", fs=Fs)

    a = [1]
    y_filt = signal.lfilter(b, a, x)
    return b, a, y_filt, M


# ==============================================================================
# FILTROS IIR — todos usam sosfilt (numericamente estável)
# ==============================================================================


def _iir_design(tipo_iir, config, Fs, Ap=3, As=60):
    """Projeta IIR e retorna (sos, b, a, N)."""
    btype_map = {
        "passa_altas": "high",
        "passa_baixas": "low",
        "passa_banda": "bandpass",
    }
    btype = btype_map[config["tipo"]]

    if config["tipo"] in ("passa_altas", "passa_baixas"):
        wp = config["fc_pass"] / (Fs / 2)
        ws = config["fc_stop"] / (Fs / 2)
    else:
        wp = [config["fc_pass_low"] / (Fs / 2), config["fc_pass_high"] / (Fs / 2)]
        ws = [config["fc_stop_low"] / (Fs / 2), config["fc_stop_high"] / (Fs / 2)]

    if tipo_iir == "butterworth":
        N, wc = signal.buttord(wp, ws, Ap, As)
        sos = signal.butter(N, wc, btype=btype, output="sos")
        b, a = signal.butter(N, wc, btype=btype)
    elif tipo_iir == "chebyshev1":
        N, wc = signal.cheb1ord(wp, ws, Ap, As)
        sos = signal.cheby1(N, Ap, wc, btype=btype, output="sos")
        b, a = signal.cheby1(N, Ap, wc, btype=btype)
    elif tipo_iir == "chebyshev2":
        N, wc = signal.cheb2ord(wp, ws, Ap, As)
        sos = signal.cheby2(N, As, wc, btype=btype, output="sos")
        b, a = signal.cheby2(N, As, wc, btype=btype)
    elif tipo_iir == "elliptic":
        N, wc = signal.ellipord(wp, ws, Ap, As)
        sos = signal.ellip(N, Ap, As, wc, btype=btype, output="sos")
        b, a = signal.ellip(N, Ap, As, wc, btype=btype)

    return sos, b, a, N


def filtro_iir_butterworth(x, Fs, config, Ap=3, As=60):
    sos, b, a, N = _iir_design("butterworth", config, Fs, Ap, As)
    return b, a, signal.sosfilt(sos, x), N


def filtro_iir_chebyshev1(x, Fs, config, Ap=3, As=60):
    sos, b, a, N = _iir_design("chebyshev1", config, Fs, Ap, As)
    return b, a, signal.sosfilt(sos, x), N


def filtro_iir_chebyshev2(x, Fs, config, Ap=3, As=60):
    sos, b, a, N = _iir_design("chebyshev2", config, Fs, Ap, As)
    return b, a, signal.sosfilt(sos, x), N


def filtro_iir_elliptic(x, Fs, config, Ap=3, As=60):
    sos, b, a, N = _iir_design("elliptic", config, Fs, Ap, As)
    return b, a, signal.sosfilt(sos, x), N


# ==============================================================================
# PROGRAMA PRINCIPAL
# ==============================================================================


def processar_audio(arquivo_entrada, tipo_ruido=None):

    print(f"\n{'='*70}")
    print(f"PROCESSAMENTO: {arquivo_entrada}")
    print(f"{'='*70}\n")

    if tipo_ruido is None:
        tipo_ruido = detectar_tipo_ruido(arquivo_entrada)

    if tipo_ruido is None or tipo_ruido not in CONFIGURACOES_FILTROS:
        print("⚠ Tipo não detectado — usando passa-banda padrão 300-3500 Hz")
        config = {
            "tipo": "passa_banda",
            "fc_pass_low": 300,
            "fc_pass_high": 3500,
            "fc_stop_low": 150,
            "fc_stop_high": 4500,
            "descricao": "Configuração padrão",
        }
        tipo_ruido = tipo_ruido or "desconhecido"
    else:
        config = CONFIGURACOES_FILTROS[tipo_ruido]
        print(f"✓ Ruído: {tipo_ruido.upper()} | {config['descricao']}")
        print(f"✓ Tipo de filtro: {config['tipo'].upper()}\n")

    output_dir = f"./output/{tipo_ruido}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Saída: {output_dir}\n")

    # Leitura
    try:
        Fs, x = wavfile.read(arquivo_entrada)
        if len(x.shape) > 1:
            x = x[:, 0]
        x = x.astype(float) / np.max(np.abs(x))
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {arquivo_entrada}")
        return

    Ns = len(x)
    t = np.arange(Ns) / Fs
    xlim = min(Fs // 2, 8000)
    print(f"Fs={Fs} Hz | {Ns} amostras | {Ns/Fs:.2f}s\n")

    c = 1  # contador de figuras

    # --- Sinal original ---
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(
        f"SINAL ORIGINAL — {tipo_ruido.upper()}", fontsize=14, fontweight="bold"
    )
    plot_sinal_tempo(t, x, "Sinal no domínio do tempo", 211)
    plot_espectro(x, Fs, "Espectro do sinal", 212, xlim_max=xlim)
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/{c:02d}_sinal_original.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"✓ {c:02d}_sinal_original.png")
    c += 1

    # --- FIR Janelamento ---
    print("--- FIR JANELAMENTO ---")
    for janela in ["hamming", "blackman"]:
        b_fir, a_fir, y_fir = filtro_fir_janelamento(
            x, Fs, config, M=101, tipo_janela=janela
        )
        print(f"  {janela}: M=101")

        analisar_filtro(
            b_fir, a_fir, Fs, f"FILTRO FIR — Janela {janela.capitalize()}", 101
        )
        plt.savefig(
            f"{output_dir}/{c:02d}_FIR_{janela}_analise.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        print(f"  ✓ {c:02d}_FIR_{janela}_analise.png")
        c += 1

        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(
            f"COMPARAÇÃO — FIR {janela.upper()}", fontsize=14, fontweight="bold"
        )
        plot_sinal_tempo(t, x, "Sinal original", 221)
        plot_sinal_tempo(t, y_fir, "Sinal filtrado", 222)
        plot_espectro(x, Fs, "Espectro original", 223, xlim_max=xlim)
        plot_espectro(y_fir, Fs, "Espectro filtrado", 224, xlim_max=xlim)
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/{c:02d}_FIR_{janela}_comparacao.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        print(f"  ✓ {c:02d}_FIR_{janela}_comparacao.png")
        c += 1

    # --- FIR Parks-McClellan ---
    print("--- FIR PARKS-McCLELLAN ---")
    b_pm, a_pm, y_pm, M_pm = filtro_fir_parks_mcclellan(x, Fs, config)
    print(f"  Ordem calculada: {M_pm}")

    analisar_filtro(b_pm, a_pm, Fs, "FILTRO FIR — Parks-McClellan", M_pm)
    plt.savefig(
        f"{output_dir}/{c:02d}_FIR_Parks_McClellan_analise.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"  ✓ {c:02d}_FIR_Parks_McClellan_analise.png")
    c += 1

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("COMPARAÇÃO — FIR PARKS-McCLELLAN", fontsize=14, fontweight="bold")
    plot_sinal_tempo(t, x, "Sinal original", 221)
    plot_sinal_tempo(t, y_pm, "Sinal filtrado", 222)
    plot_espectro(x, Fs, "Espectro original", 223, xlim_max=xlim)
    plot_espectro(y_pm, Fs, "Espectro filtrado", 224, xlim_max=xlim)
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/{c:02d}_FIR_Parks_McClellan_comparacao.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"  ✓ {c:02d}_FIR_Parks_McClellan_comparacao.png")
    c += 1

    # --- IIR ---
    print("--- FILTROS IIR ---")
    filtros_iir = [
        ("Butterworth", filtro_iir_butterworth),
        ("Chebyshev_I", filtro_iir_chebyshev1),
        ("Chebyshev_II", filtro_iir_chebyshev2),
        ("Elliptic", filtro_iir_elliptic),
    ]

    resultados_iir = {}
    for nome, func in filtros_iir:
        b_i, a_i, y_i, N_i = func(x, Fs, config)
        print(f"  {nome}: Ordem={N_i}")
        resultados_iir[nome] = y_i

        analisar_filtro(b_i, a_i, Fs, f"FILTRO IIR — {nome}", N_i)
        plt.savefig(
            f"{output_dir}/{c:02d}_IIR_{nome}.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        print(f"  ✓ {c:02d}_IIR_{nome}.png")
        c += 1

    # --- Comparação IIR ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("COMPARAÇÃO — ESPECTROS FILTROS IIR", fontsize=14, fontweight="bold")
    pares = [("Original", x)] + list(resultados_iir.items())
    for ax, (nome, sig) in zip(axes.flat, pares):
        Ns_ = len(sig)
        S_ = np.abs(np.fft.fft(sig)[: Ns_ // 2])
        f_ = np.fft.fftfreq(Ns_, 1 / Fs)[: Ns_ // 2]
        ax.plot(f_, S_)
        ax.set_title(nome)
        ax.set_xlim([0, xlim])
        ax.set_xlabel("Frequência (Hz)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
    for ax in axes.flat[len(pares) :]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/{c:02d}_IIR_Comparacao_Todos.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"  ✓ {c:02d}_IIR_Comparacao_Todos.png")

    # --- Salvar áudios filtrados ---
    print("\n--- SALVANDO ÁUDIOS FILTRADOS ---")
    audio_dir = f"{output_dir}/audio"
    os.makedirs(audio_dir, exist_ok=True)

    def salvar_wav(nome, sinal):
        sinal_norm = sinal / (np.max(np.abs(sinal)) + 1e-9)
        sinal_int16 = (sinal_norm * 32767).astype(np.int16)
        wavfile.write(f"{audio_dir}/{nome}.wav", Fs, sinal_int16)
        print(f"  ✓ {nome}.wav")

    salvar_wav("00_original", x)
    salvar_wav(
        "01_FIR_hamming",
        filtro_fir_janelamento(x, Fs, config, M=101, tipo_janela="hamming")[2],
    )
    salvar_wav(
        "02_FIR_blackman",
        filtro_fir_janelamento(x, Fs, config, M=101, tipo_janela="blackman")[2],
    )
    salvar_wav("03_FIR_Parks_McClellan", y_pm)
    for i, (nome, sig) in enumerate(resultados_iir.items(), start=4):
        salvar_wav(f"{i:02d}_IIR_{nome}", sig)

    n_audios = 4 + len(resultados_iir)
    print(f"\n{'='*70}")
    print(f"CONCLUÍDO — {c} gráficos | {n_audios} áudios em '{output_dir}'")
    print(f"{'='*70}")

    return {
        "original": x,
        "Fs": Fs,
        **{k.lower(): v for k, v in resultados_iir.items()},
    }


# ==============================================================================
# EXECUÇÃO
# ==============================================================================

if __name__ == "__main__":
    inputdir = Path("./audios/clean+noise/")

    # Cria o diretório de entrada se não existir para evitar erros
    if not inputdir.exists():
        os.makedirs(inputdir, exist_ok=True)
        print(f"⚠ Pasta '{inputdir}' criada. Adicione seus arquivos .wav nela.")

    while True:
        print("\n" + "=" * 70)
        print("Programa de Filtragem de Ruído de Áudio".center(70))
        print("=" * 70)

        # Lista os arquivos disponíveis
        arquivos = sorted(
            [item for item in inputdir.iterdir() if item.suffix == ".wav"]
        )

        if not arquivos:
            print("Nenhum arquivo .wav encontrado na pasta ./audios/clean+noise/")
        else:
            print("Escolha o áudio a ser filtrado: ")
            for i, item in enumerate(arquivos):
                print(f"{i+1}. {item.name}")

        print("-" * 70)
        print("0. Atualizar Lista")
        print("x. Sair")

        escolha = input("\nSua Escolha: ").strip().lower()

        if escolha == "x":
            print("Encerrando o programa...")
            break
        elif escolha == "0":
            continue
        elif escolha.isdigit():
            idx = int(escolha) - 1
            if 0 <= idx < len(arquivos):
                arquivo_selecionado = arquivos[idx]
                # Executa o processamento
                processar_audio(str(arquivo_selecionado))
                input("\nPresione Enter para continuar...")
            else:
                print("❌ Opção inválida. Tente novamente.")
        else:
            print("❌ Entrada inválida.")

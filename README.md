# Filtro de Ruídos Digital (FIR & IIR) 🎧

Este projeto foi desenvolvido para a disciplina de **Processamento Digital de Sinais** da **Universidade Federal do Oeste da Bahia (UFOB)**, Campus Bom Jesus da Lapa. Sob Orientação do Professor Elias Guimarães.

O objetivo do software é realizar a filtragem de ruídos específicos em arquivos de áudio utilizando diferentes técnicas de design de filtros digitais, comparando as respostas de filtros FIR (Finite Impulse Response) e IIR (Infinite Impulse Response).

## 🚀 Funcionalidades

* **Detecção Automática de Ruído:** O programa identifica o tipo de ruído pelo nome do arquivo (ex: *vacuum cleaner*, *typing*, *babble*).
* **Múltiplas Arquiteturas de Filtro:**
* **FIR:** Janelamento (Hamming, Blackman) e Parks-McClellan (Remez).
* **IIR:** Butterworth, Chebyshev Tipo I e II, e Elíptico.


* **Análise Visual Completa:** Gera gráficos de magnitude (linear e dB), fase, atraso de grupo, diagrama de polos e zeros e resposta ao impulso.
* **Comparativo de Espectros:** Compara o espectro de frequência do sinal original versus o filtrado para validar a atenuação do ruído.
* **Exportação de Áudio:** Salva os resultados processados em formato `.wav` para avaliação auditiva.

## 📂 Estrutura de Arquivos

Para que o programa funcione corretamente, organize os arquivos da seguinte forma:

```text
trabalho_pds/
├── audios/
│   └── clean+noise/        # Coloque seus arquivos .wav aqui
├── output/                 # Gerado automaticamente com os resultados
├── noise_filter.py         # Script principal
├── pyproject.toml          # Dependências do projeto
└── README.md

```

### Áudios Suportados (Padrão)

O sistema possui configurações otimizadas para os seguintes arquivos (já incluídos ou que podem ser adicionados):

1. `clnsp10_VacuumCleaner.wav` (Aspirador de pó)
2. `clnsp11_bemtevi.wav` (Canto de pássaro/Ruído de fundo)
3. `clnsp12_hiss.wav` (Ruído branco/Hiss)
4. `clnsp1_airconditioner.wav` (Ar condicionado)
5. ... e outros como `airport`, `babble`, `copymachine`, `munching`, `typing`.

## 🛠️ Requisitos e Instalação

O projeto utiliza **Python 3.14+** (conforme `pyproject.toml`).

1. **Instale as dependências:**
```bash
pip install numpy matplotlib scipy

```


*(Ou utilize o gestor de sua preferência com o `pyproject.toml` fornecido)*.
2. **Execute o programa:**
```bash
python noise_filter.py

```



## 📊 Processo de Filtragem

Ao selecionar um áudio, o programa executa o seguinte fluxo:

1. **Leitura do Sinal:** Normalização do áudio e extração de metadados.
2. **Cálculo dos Coeficientes:** * Para **IIR**, utiliza-se a implementação via **SOS** (Second-Order Sections) para garantir estabilidade numérica.
* Para **FIR**, as ordens são calculadas automaticamente (Kaiser) ou definidas por janelamento.


3. **Geração de Relatórios:** Os gráficos são salvos na pasta `./output/[tipo_do_ruido]/`.

> *Exemplo: O diagrama de Polos e Zeros gerado ajuda a verificar a estabilidade do filtro IIR e a linearidade de fase dos filtros FIR.*

## 🧑‍💻 Autores

* **Eder Renato** - [EderRenato](https://github.com/EderRenato)
* **Keylla Kaylla** - [KeyllaK](https://github.com/KeyllaK)
* **Luis Felipe** - [luisfbsilva](https://github.com/luisfbsilva)
* **Instituição:** UFOB - Engenharia Elétrica.

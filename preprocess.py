import os  # Importa o módulo os para funcionalidades do sistema operacional (como criar diretórios).
import cv2  # Importa o OpenCV (cv2) para processamento de vídeo.
import numpy as np  # Importa o NumPy para operações numéricas, especialmente manipulação de arrays.
import librosa  # Importa o librosa para análise e manipulação de áudio.
import matplotlib.pyplot as plt  # Importa o Matplotlib para plotar e salvar imagens.
from tqdm import tqdm  # Importa tqdm para exibir barras de progresso em loops.
from librosa import feature as audio  # Importa o módulo de features do librosa, renomeado como 'audio' para conveniência.


"""
Estrutura do conjunto de dados AVLips:
AVLips
├── 0_real
├── 1_fake
└── wav
    ├── 0_real
    └── 1_fake
"""

############ Parâmetro personalizado ##############
N_EXTRACT = 10   # Número de segmentos de imagem a serem extraídos de cada vídeo.
WINDOW_LEN = 5   # Número de frames consecutivos em cada segmento.
MAX_SAMPLE = 100  # Número máximo de vídeos a serem processados (para teste/limitar tempo de execução).

audio_root = "./AVLips/wav"  # Caminho para os arquivos de áudio.
video_root = "./AVLips"      # Caminho para os arquivos de vídeo.
output_root = "./datasets/AVLips"  # Caminho para salvar os dados pré-processados.
############################################

labels = [(0, "0_real"), (1, "1_fake")] # Lista de rótulos: (dois elementos. Cada elemento é uma tupla).

def get_spectrogram(audio_file): # Função para gerar e salvar um Mel-espectrograma.
    data, sr = librosa.load(audio_file)  # Carrega o arquivo de áudio; data = áudio, sr = taxa de amostragem
    mel = librosa.power_to_db(audio.melspectrogram(y=data, sr=sr), ref=np.min)  # Computa o Mel-espectrograma e converte para dB.
    plt.imsave("./temp/mel.png", mel)  # Salva o espectrograma como uma imagem PNG.


def run():  # Função principal para pré-processar o conjunto de dados.
    i = 0  # Contador para o número de conjuntos de dados processados.
    for label, dataset_name in labels:  # Itera pelos rótulos ("0_real" e "1_fake").
        if not os.path.exists(dataset_name):  # Verifica se o diretório de saída para este conjunto de dados existe.
            os.makedirs(f"{output_root}/{dataset_name}", exist_ok=True)  # Cria se não existir.

        if i == MAX_SAMPLE:  # Verifica se o limite máximo de amostras foi alcançado.
            break  # Interrompe o loop se o limite for alcançado.
        root = f"{video_root}/{dataset_name}"  # Constrói o caminho para os arquivos de vídeo do conjunto de dados atual.
        video_list = os.listdir(root)  # Obtém uma lista de todos os arquivos de vídeo no diretório do conjunto de dados.
        print(f"Processando {dataset_name}...")  # Imprime uma mensagem indicando qual conjunto de dados está sendo processado.
        for j in tqdm(range(len(video_list))):  # Itera pela lista de arquivos de vídeo, usando tqdm para uma barra de progresso.
            v = video_list[j]  # Obtém o nome do arquivo do vídeo atual.
            # Carrega o vídeo
            video_capture = cv2.VideoCapture(f"{root}/{v}")  # Abre o arquivo de vídeo usando OpenCV.
            fps = video_capture.get(cv2.CAP_PROP_FPS)  # Obtém os quadros por segundo (fps) do vídeo.
            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # Obtém o número total de quadros no vídeo.

            # seleciona 10 pontos de partida dos quadros
            frame_idx = np.linspace( # Gera índices igualmente espaçados para extrair segmentos do vídeo.
                0,
                frame_count - WINDOW_LEN - 1,
                N_EXTRACT,
                endpoint=True,
                dtype=np.uint8,
            ).tolist()
            frame_idx.sort()  # Ordena os índices.
            # quadros selecionados
            frame_sequence = [ # Expande os índices para obter a sequência de quadros a serem extraídos para cada segmento.
                i for num in frame_idx for i in range(num, num + WINDOW_LEN)
            ]
            frame_list = []  # Lista para armazenar os quadros extraídos.
            current_frame = 0  # Índice do quadro atual sendo lido.
            while current_frame <= frame_sequence[-1]:  # Loop pelos quadros do vídeo.
                ret, frame = video_capture.read()  # Lê um quadro do vídeo.
                if not ret:  # Verifica se o quadro foi lido com sucesso.
                    print(f"Erro ao ler o quadro {v}: {current_frame}")  # Imprime uma mensagem de erro se o quadro não foi lido.
                    break  # Interrompe o loop se ocorreu um erro.
                if current_frame in frame_sequence:  # Verifica se o quadro atual é um dos quadros a serem extraídos.
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # Converte o quadro de BGR (padrão do OpenCV) para RGBA.
                    frame_list.append(cv2.resize(frame, (500, 500)))  # Redimensiona o quadro e o adiciona à lista.
                current_frame += 1  # Incrementa o índice do quadro atual.
            video_capture.release()  # Fecha o arquivo de vídeo.

            # Carrega o áudio
            name = v.split(".")[0]  # Extrai o nome (sem extensão) do nome do arquivo de vídeo.
            a = f"{audio_root}/{dataset_name}/{name}.wav"  # Constrói o caminho para o arquivo de áudio correspondente.

            group = 0  # Contador para o grupo de imagens de saída.
            get_spectrogram(a)  # Gera e salva o Mel-espectrograma para o áudio.
            mel = plt.imread("./temp/mel.png") * 255  # Carrega o Mel-espectrograma; a escala por 255 parece converter para uint8.
            mel = mel.astype(np.uint8)  # Converte o espectrograma para o tipo de dado uint8 (inteiro sem sinal de 8 bits).
            mapping = mel.shape[1] / frame_count  # Calcula um fator de escala para mapear o tempo do áudio para os quadros do vídeo.
            for i in range(len(frame_list)):  # Itera pelos quadros de vídeo extraídos.
                idx = i % WINDOW_LEN  # Obtém o índice dentro da janela atual.
                if idx == 0:  # Processa apenas o primeiro quadro em cada janela.
                    try:
                        begin = np.round(frame_sequence[i] * mapping)  # Calcula o índice inicial para o segmento do Mel-espectrograma.
                        end = np.round((frame_sequence[i] + WINDOW_LEN) * mapping)  # Calcula o índice final para o segmento do Mel-espectrograma.
                        sub_mel = cv2.resize( # Redimensiona o segmento do Mel-espectrograma para corresponder ao tamanho do quadro do vídeo.
                            (mel[:, int(begin) : int(end)]), (500 * WINDOW_LEN, 500)
                        )
                        x = np.concatenate(frame_list[i : i + WINDOW_LEN], axis=1)  # Concatena os quadros na janela atual horizontalmente.
                        x = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)  # Concatena o Mel-espectrograma e os quadros do vídeo verticalmente.
                        plt.imsave(  # Salva a imagem combinada.
                            f"{output_root}/{dataset_name}/{name}_{group}.png", x
                        )
                        group = group + 1  # Incrementa o contador de grupos.
                    except ValueError:  # Lida com possíveis erros (por exemplo, se os índices estiverem fora dos limites).
                        print(f"ValueError: {name}")  # Imprime uma mensagem de erro.
                        continue  # Pula para a próxima iteração.
        i += 1  # Incrementa o contador de conjuntos de dados.


if __name__ == "__main__":  # Verifica se o script é o programa principal sendo executado (não importado como um módulo).
    if not os.path.exists(output_root):  # Verifica se o diretório de saída existe.
        os.makedirs(output_root, exist_ok=True)  # Cria se não existir.
    if not os.path.exists("./temp"):  # Verifica se o diretório temporário existe.
        os.makedirs("./temp", exist_ok=True)  # Cria se não existir.
    run()  # Executa a função principal de pré-processamento.
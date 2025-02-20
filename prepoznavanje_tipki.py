import os
import numpy as np
from pydub import AudioSegment
import librosa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# Postavite putanju za ffmpeg (ako još nije postavljeno)
AudioSegment.ffmpeg = "C:/Users/Milixo/Desktop/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe"


# Funkcija za učitavanje svih snimki iz određenog foldera
def load_data(data_dir):
    audio_data = []
    labels = []

    # Prolazak kroz sve foldere u mapi
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)

        if os.path.isdir(folder_path):  # Provjeravamo je li folder
            print(f"Obrađujem folder: {folder}")  # Ispis za praćenje obrade
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if file_name.endswith('.wav'):  # Provjeravamo samo WAV datoteke
                    print(f"Obrađujem datoteku: {file_name}")  # Ispis imena datoteke
                    # Učitavanje zvučne datoteke
                    audio = AudioSegment.from_wav(file_path)

                    # Ekstrakcija značajki (MFCC)
                    audio_data.append(extract_features(audio))
                    labels.append(folder)  # Koristimo naziv foldera kao labelu (tipku)

    return np.array(audio_data), np.array(labels)


# Funkcija za ekstrakciju MFCC značajki
def extract_features(audio, sr=22050, n_mfcc=13):
    """
    Ekstraktira MFCC značajke iz audio datoteke.

    :param audio: Pydub AudioSegment objekt
    :param sr: Sample rate (učestalost uzorkovanja)
    :param n_mfcc: Broj MFCC koeficijenata koje želimo izdvojiti
    :return: Srednje vrijednosti MFCC koeficijenata za cijeli signal
    """
    # Pretvaranje AudioSegment objekta u numpy array
    samples = np.array(audio.get_array_of_samples())

    # Normalizacija audio podataka na interval [-1, 1]
    samples = samples / (2 ** 15)  # Audio podaci su obično u opsegu [-32768, 32767] za 16-bitne wav datoteke

    # Pretvaranje u float32
    samples = samples.astype(np.float32)

    # Ekstrakcija MFCC značajki koristeći librosa
    mfccs = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=n_mfcc)

    # Povrat samo srednje vrijednosti MFCC koeficijenata kroz sve vremenske korake
    return np.mean(mfccs, axis=1)


# Putanja do mape s podacima
data_dir = "data_set_tipkovnice"  # Provjerite putanju!

# Učitavanje podataka
audio_data, labels = load_data(data_dir)

# Ispisivanje podataka za sve tipke
for i, label in enumerate(np.unique(labels)):
    print(f"Broj snimki za tipku {label}: {np.count_nonzero(labels == label)}")

# Ispis prvih nekoliko značajki za svaku tipku
for i, label in enumerate(np.unique(labels)):
    print(f"\nPrve značajke za tipku {label}:")
    indices = np.where(labels == label)[0]
    print(audio_data[indices[0]])  # Ispis prvih značajki za prvu snimku te tipke

# Korak 3: Treniranje modela (SVM)
# Podjela podataka na trening i testni skup
X_train, X_test, y_train, y_test = train_test_split(audio_data, labels, test_size=0.2, random_state=42)

# Kreiranje i treniranje SVM modela
model = SVC(kernel='linear')  # Možete isprobati i druge vrste kernela (npr. 'rbf', 'poly')
model.fit(X_train, y_train)

# Predviđanje na testnim podacima
y_pred = model.predict(X_test)

# Evaluacija modela
print("\nEvaluacija modela:")
print(classification_report(y_test, y_pred))

# Ispisivanje točnosti na cijelom testnom skupu
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTočnost modela: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(labels))
disp.plot(cmap='Blues')
plt.title("Matrica konfuzije")
plt.savefig("confusion_matrix.png")  # Spremi graf
plt.show()

# Kros-validacija
cross_val_scores = cross_val_score(model, audio_data, labels, cv=5)
print(f"\nProsječna preciznost tijekom 5-fold kros-validacije: {cross_val_scores.mean():.4f}")

# **Vizualizacija distribucije MFCC značajki za svaku tipku**
for label in np.unique(labels):
    indices = np.where(labels == label)[0]
    plt.figure(figsize=(12, 8))

    # Kreiraj colormap (tab20 daje 20 različitih boja)
    colors = plt.cm.tab20(np.linspace(0, 1, len(indices)))  # tab20 daje 20 boja

    # Prikaz svih uzoraka za svaku tipku
    for i in range(len(indices)):  # Prikaz svih uzoraka za svaku tipku
        plt.plot(audio_data[indices[i]], label=f"Snimak {i + 1}", color=colors[i])  # Dodaj boju iz colormap
    plt.title(f"Distribucija MFCC značajki za tipku '{label}'")
    plt.xlabel("MFCC koeficijent")
    plt.ylabel("Vrijednost")
    # Legenda dolje
    plt.legend() #legenda
    plt.savefig(f"mfcc_distribution_all_{label}.png")  # Spremi graf
    plt.show()

# **Vizualizacija točnosti modela**
correct = (y_test == y_pred)
plt.figure(figsize=(8, 6))
plt.bar(["Točno predviđeno", "Netočno predviđeno"], [np.sum(correct), len(correct) - np.sum(correct)],
        color=['green', 'red'])
plt.title("Pregled rezultata predikcije")
plt.ylabel("Broj primjera")
plt.savefig("prediction_results.png")  # Spremi graf
plt.show()

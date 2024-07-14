import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from wordcloud import WordCloud
import joblib
import os
import io
from docx import Document
import datetime
import base64

# File paths
df_fake_path = "Data/Fake_fix.xlsx"
df_true_path = "Data/True_2.csv"

# Function to clean the text
def clean_text(text):
    text = str(text)  # Pastikan teks diubah menjadi string terlebih dahulu
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Function to check login
def check_login(username, password):
    return username == "admin" and password == "123"

# Function to save model
def save_model(model, model_name):
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    try:
        joblib.dump(model, model_path)
        st.write(f"Model saved as {model_path}")
    except Exception as e:
        st.write(f"Failed to save {model_name}: {e}")

# Function to load model
def load_model(model_name):
    model_path = os.path.join("models", f"{model_name}.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.write(f"Model {model_name} loaded.")
        return model
    else:
        st.write(f"Model {model_name} not found.")
        return None

# Function for document input
def document_input(vectorization, models, xv_train, y_train):
    st.header("Input Dokumen")
    st.write("Anda dapat memasukkan teks berita untuk dianalisis di sini.")

    uploaded_file = st.file_uploader("Unggah file Word (.docx)", type="docx")

    if uploaded_file is not None:
        docx = io.BytesIO(uploaded_file.read())
        document = Document(docx)
        full_text = [para.text for para in document.paragraphs]
        news_input = '\n'.join(full_text)
        cleaned_text = clean_text(news_input)

        model_choice = st.sidebar.selectbox("Pilih model", list(models.keys()))
        loaded_model = load_model(model_choice)
        
        if loaded_model:
            selected_model = loaded_model
        else:
            selected_model = models[model_choice]
            selected_model.fit(xv_train, y_train)

        xv_input = vectorization.transform([cleaned_text])
        pred = selected_model.predict(xv_input)

        if pred[0] == 0:
            st.write("Hasil Deteksi: Berita Hoaks")
        else:
            st.write("Hasil Deteksi: Berita Real")

# Function to display About information
def display_about():
    st.title("Tentang Aplikasi Deteksi Hoax")

    col1, col2 = st.columns(2)

    with col1:
        st.write("""
            ### Mengapa menggunakan aplikasi ini?
            - **Prediksi Berita Hoax:** Aplikasi ini dapat memprediksi apakah suatu berita merupakan hoax atau tidak.
            - **Bantu Tenaga Media:** Membantu tenaga media dalam mengidentifikasi berita hoax.
            - **Kesadaran Publik:** Membantu masyarakat untuk mengetahui apakah berita yang mereka baca benar atau tidak.
            - **Pemantauan Berita:** Membantu masyarakat dalam memantau kualitas berita yang mereka konsumsi.
        """)

    with col2:
        st.write("""
            ### Mengapa menggunakan dataset ini?
            - **Keterkaitan Faktor Risiko Berita Hoax:** Memahami keterkaitan berbagai faktor risiko yang terkait dengan berita hoax.
            - **Prediksi dan Diagnosis:** Membantu dalam prediksi dan diagnosis berita hoax.
            - **Informasi Penting:** Memberikan informasi penting terkait dengan penyebaran berita hoax.
            - **Aplikasi Dunia Nyata:** Dataset ini memiliki aplikasi nyata dalam dunia jurnalisme dan media.
        """)

    col3, col4 = st.columns(2)

    with col3:
        st.write("""
            ### Tentang Dataset Deteksi Hoax
            - **Penggunaan Dataset untuk Melatih Model:** Dataset ini digunakan untuk melatih model deteksi hoax.
            - **Tautan Dataset Deteksi Hoax:** [Klik Di Sini](https://www.kaggle.com/code/therealsampat/fake-news-detection/notebook)
        """)

    with col4:
        st.write("""
            ### Mengapa menggunakan dataset ini?
            - **Keterkaitan Faktor Risiko Berita Hoax:** Memahami keterkaitan berbagai faktor risiko yang terkait dengan berita hoax.
            - **Prediksi dan Diagnosis:** Membantu dalam prediksi dan diagnosis berita hoax.
            - **Informasi Penting:** Memberikan informasi penting terkait dengan penyebaran berita hoax.
            - **Aplikasi Dunia Nyata:** Dataset ini memiliki aplikasi nyata dalam dunia jurnalisme dan media.
        """)

    st.write("")
    
    st.write("## Tim Kami")
    
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        img = open("assets/Gina.jpg", "rb")
        base64_img = base64.b64encode(img.read()).decode()
        img.close()
        st.markdown(
            f'<img src="data:image/png;base64,{base64_img}" style="border-radius: 10%; width: 400px; height: 400px;"><p style="margin-bottom: 20px;"></p>',
            unsafe_allow_html=True,
        )
        st.write("""
            **Ginanti Riski** adalah seorang mahasiswi yang bertanggung jawab atas pengumpulan dan analisis data, pengembangan kode, desain antarmuka pengguna, serta implementasi dan hosting aplikasi ini.
        """)

if 'history_data' not in st.session_state:
    st.session_state.history_data = []

def display_history():
    st.header("History")
    st.write("Halaman Tabel")

    if len(st.session_state.history_data) == 0:
        st.write("Tidak ada history prediksi")
    else:
        st.write("Tabel History Prediksi")
        history_df = pd.DataFrame(st.session_state.history_data, columns=["Berita", "Hasil", "Waktu Deteksi"])
        st.write(history_df)

        st.write("Pilih nomor entri untuk dihapus")
        entry_to_delete = st.selectbox("Nomor entri:", range(1, len(st.session_state.history_data) + 1))

        if st.button("Hapus entri"):
            del st.session_state.history_data[entry_to_delete - 1]
            st.experimental_rerun()

        if st.button("Hapus semua entri"):
            st.session_state.history_data.clear()
            st.experimental_rerun()

# Function to display detection history
history_df = pd.DataFrame(columns=["text", "class", "detected_at"])

def multiple_predict(vectorization, models, xv_train, y_train):
    st.header("Multiple Predict")
    st.write("Masukkan beberapa teks berita untuk dianalisis:")

    texts_input = st.text_area("Teks Berita:", height=300)

    model_choice = st.sidebar.selectbox("Pilih model", list(models.keys()))

    if st.button("Deteksi"):
        paragraphs = [text.strip() for text in texts_input.split("\n\n") if text.strip()]

        results_table = []

        for i, paragraph in enumerate(paragraphs, start=1):
            cleaned_text = clean_text(paragraph)
            xv_input = vectorization.transform([cleaned_text])

            loaded_model = load_model(model_choice)

            if loaded_model:
                selected_model = loaded_model
            else:
                selected_model = models[model_choice]
                selected_model.fit(xv_train, y_train)

            pred = selected_model.predict(xv_input)
            pred_label = "Real" if pred[0] == 1 else "Hoax"

            results_table.append([f"Berita {i}", pred_label])

        st.write(pd.DataFrame(results_table, columns=["Berita", "Hasil"]))

        st.session_state.history_data.extend(results_table)

# Main function for Streamlit app
def main():
    st.title("Deteksi Berita Hoax di Indonesia")

    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Sidebar CSS customization
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #f0f0f0;
        }
        .sidebar .sidebar-content .sidebar-close-btn {
            color: #000;
        }
        .sidebar .sidebar-content .stSelectbox .stOption:hover {
            background-color: #4caf4f !important;
        }
        .sidebar .sidebar-content .stSelectbox label {
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Login section
    if not st.session_state.logged_in:
        st.title("Login")
        username = st.text_input("Username:")
        password = st.text_input("Password:", type="password")

        if st.button("Login"):
            if check_login(username, password):
                st.success("Login Berhasil!")
                st.session_state.logged_in = True
            else:
                st.warning("Username atau Password salah!")

    # Main content section after successful login
    if st.session_state.logged_in:
        # Load data function
        @st.cache_data
        def load_data():
            try:
                df_fake = pd.read_excel(df_fake_path, engine='openpyxl')
                df_true = pd.read_csv(df_true_path)
                return df_fake, df_true
            except Exception as e:
                st.error(f"Error reading datasets: {e}")
                return None, None

        # Load data
        df_fake, df_true = load_data()

        if df_fake is not None and df_true is not None:
            # Add class column
            df_fake["class"] = 0
            df_true["class"] = 1

            # Remove unnecessary columns
            df_fake = df_fake.drop(["title", "subjek", "date"], axis=1)
            df_true = df_true.drop(["title", "subjek", "date"], axis=1)

            # Manual testing data
            df_fake_manual_testing = df_fake.tail(10).copy()
            df_true_manual_testing = df_true.tail(10).copy()

            df_fake.drop(df_fake.tail(10).index, inplace=True)
            df_true.drop(df_true.tail(10).index, inplace=True)

            df_fake_manual_testing["class"] = 0
            df_true_manual_testing["class"] = 1

            df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
            df_manual_testing.to_csv("manual_testing.csv", index=False)

            # Merge dataframes
            df_merge = pd.concat([df_fake, df_true], axis=0)

            # Clean text
            df_merge["text"] = df_merge["text"].astype(str).apply(clean_text)

            # Define dependent and independent variables
            x = df_merge["text"]
            y = df_merge["class"]

            # Split data
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

            # Vectorization
            vectorization = TfidfVectorizer(max_features=56300)
            xv_train = vectorization.fit_transform(x_train)
            xv_test = vectorization.transform(x_test)

            # Models
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
            }

            # Sidebar menu
            menu = st.sidebar.selectbox("Menu", ["Pemilihan Model", "Deteksi Berita", "Visualisasi", "EDA", "Lainnya"])

            if menu == "Pemilihan Model":
                st.header("Pemilihan Model")
                model_choice = st.sidebar.selectbox("Pilih model", list(models.keys()))
                loaded_model = load_model(model_choice)

                if loaded_model:
                    selected_model = loaded_model
                else:
                    selected_model = models[model_choice]
                    selected_model.fit(xv_train, y_train)

                if st.button("Save Model"):
                    save_model(selected_model, model_choice)

            elif menu == "Deteksi Berita":
                st.header("Deteksi Berita")
                model_choice = st.sidebar.selectbox("Pilih model", list(models.keys()))
                loaded_model = load_model(model_choice)

                if loaded_model:
                    selected_model = loaded_model
                else:
                    selected_model = models[model_choice]
                    selected_model.fit(xv_train, y_train)

                news_input = st.text_area("Masukkan berita:")
                if st.button("Deteksi"):
                    xv_input = vectorization.transform([clean_text(news_input)])
                    pred = selected_model.predict(xv_input)
                    class_label = "Hoax" if pred[0] == 0 else "Real"
                    detection_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    st.session_state.history_data.append([news_input, class_label, detection_time])
                    st.write("Hasil Deteksi: Berita " + class_label)

            elif menu == "Visualisasi":
                st.header("Visualisasi")
                st.subheader("Word Cloud")

                news_input = st.text_area("Masukkan berita:", key="news_input")

                if news_input:
                    st.write("Tekan tombol 'Visualisasi' untuk melihat hasil visualisasi")
                    visualize_button = st.button("Visualisasi", key="visualize_button")

                    cleaned_text = clean_text(news_input)  # Define cleaned_text here

                    if visualize_button:
                        # Wordcloud
                        if len(cleaned_text) > 0:
                            wordcloud = WordCloud(width=800, height=400, max_words=100).generate(cleaned_text)

                            model_choice = st.sidebar.selectbox("Pilih model", list(models.keys()), key="model_choice")
                            loaded_model = load_model(model_choice)

                            if loaded_model:
                                selected_model = loaded_model
                            else:
                                selected_model = models[model_choice]
                                selected_model.fit(xv_train, y_train)

                            xv_input = vectorization.transform([cleaned_text])
                            pred = selected_model.predict(xv_input)

                            if pred[0] == 0:
                                title = "Wordcloud Berita Hoax"
                            else:
                                title = "Wordcloud Berita Real"

                            st.subheader(title)
                            plt.figure(figsize=(10, 5))
                            plt.imshow(wordcloud, interpolation="bilinear")
                            plt.axis("off")
                            st.pyplot(plt)
                        else:
                            st.write("Tidak ada kata dalam inputan. Silakan masukkan inputan yang valid.")
                else:
                    st.write("Masukkan inputan terlebih dahulu untuk melihat visualisasi berita")

                # Confusion Matrix
                st.subheader("Confusion Matrix")
                model_choice = st.sidebar.selectbox("Pilih model", list(models.keys()))
                loaded_model = load_model(model_choice)

                if loaded_model:
                    selected_model = loaded_model
                else:
                    selected_model = models[model_choice]
                    selected_model.fit(xv_train, y_train)

                xv_input = vectorization.transform([cleaned_text])
                pred = selected_model.predict(xv_input)
                y_pred = [pred[0]]

                cm = confusion_matrix([0], y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, cmap="Blues")
                plt.xlabel("Predicted labels")
                plt.ylabel("True labels")
                plt.title("Confusion Matrix")
                st.pyplot(plt)

            elif menu == "EDA":
                st.header("Exploratory Data Analysis (EDA)")
                st.subheader("Dataframe")

                news_input = st.text_area("Masukkan berita:")

                if news_input:
                    st.write("Tekan tombol 'EDA' untuk melakukan analisis")
                    eda_button = st.button("EDA")

                    if eda_button:
                        cleaned_text = clean_text(news_input)
                        class_value = 0 if "fake" in news_input.lower() else 1  # assume fake if "fake" is in the text, otherwise real

                        df_input = pd.DataFrame({"text": [cleaned_text], "class": [class_value]})

                        # merge with existing dataframe (if any)
                        if 'df_merge' in st.session_state:
                            df_merge = st.session_state.df_merge
                            df_merge = pd.concat([df_merge, df_input])
                        else:
                            df_merge = df_input

                        st.session_state.df_merge = df_merge

                        st.subheader("Dataframe")
                        st.dataframe(df_merge.head())

                        st.subheader("Distribusi Kelas")
                        class_counts = df_merge["class"].value_counts()
                        st.bar_chart(class_counts)

                        st.subheader("Panjang Teks")
                        df_merge["text_length"] = df_merge["text"].apply(lambda x: len(x))
                        fig, ax = plt.subplots()
                        ax.hist(df_merge["text_length"], bins=50)
                        st.pyplot(fig)
                else:
                    st.write("Masukkan inputan terlebih dahulu untuk melakukan analisis")


            elif menu == "Lainnya":
                st.header("Lainnya")
                sub_menu = st.sidebar.selectbox("Pilihan Lainnya", ["About", "History Deteksi", "Multiple Predict", "Upload File"])

                if sub_menu == "About":
                    display_about()

                elif sub_menu == "History Deteksi":
                    display_history()

                elif sub_menu == "Multiple Predict":
                    multiple_predict(vectorization, models, xv_train, y_train)

                elif sub_menu == "Upload File":
                    document_input(vectorization, models, xv_train, y_train)

            # Logout button
            if st.sidebar.button("Logout"):
                st.session_state.logged_in = False
                st.success("Anda telah logout.")

        else:
            st.error("Error loading datasets. Please check file paths.")

# Run the app
if __name__ == "__main__":
    main()
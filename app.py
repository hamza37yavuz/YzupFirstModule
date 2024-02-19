# Github Repository: https://github.com/hamza37yavuz/YzupProje


import streamlit as st
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class App:
    def __init__(self):
        self.datasets = {
            "Breast Cancer": pd.read_csv("data.csv")
        }
        self.df = pd.read_csv("data.csv")
        # Drop
        self.selectedDataset = None
        self.params = None
        self.X = None
        self.y = None
        self.X_train, self.X_test, self.Y_train, self.Y_test = None, None, None, None
        self.model = KNeighborsClassifier()
        self.modelnames = ["KNN", "Support Vector Machine", "Naïve Bayes"]
        self.selectedModels = None
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
    # OTHER FUNCTIONS
    def SelectedCorrelation(self, selectedColumns):
        # Secilen sutunlar arasindaki korelasyon matrisini hesaplama

        selecteDf = self.df[list(selectedColumns)] 
        corrMatrix = selecteDf.corr()

        # Korelasyon matrisini gorsellestirme
        plt.figure(figsize=(10, 8))
        sns.heatmap(corrMatrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Secilen Sutunlarin Korelasyon Matrisi")
        st.pyplot()
        st.write(f"Bu matrise gore 0.90 a esit ve 0.90 dan buyuk olan sutunlar ayni durumu temsil etmektedir.")
        self.X.drop(columns=selectedColumns, axis = 1, inplace=True)
        st.write(f"0.90'dan buyuk olan sutunlardan birisi drop edilmistir. Yeni kolon sayisi: {self.X.shape[1]}")
        st.write("### Veri Seti Ilk 2 Satiri:")
        st.write(self.X.head(2)) 
    
    def detectOutlier(self, num_col_name, q1=0.10, q3=0.90):
        """
        The function performs an outlier analysis within numerical columns of a given dataframe. 
        It operates by iterating through the numerical variables using a 'for' loop. 
        The values of q1 and q3 are kept low to avoid disrupting the natural structure of the data. These values can be optionally adjusted.
        """
        quartile1 = self.df[num_col_name].quantile(q1)
        quartile3 = self.df[num_col_name].quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range

        if self.df[(self.df[num_col_name] > up_limit) | (self.df[num_col_name] < low_limit)].any(axis=None):
            return True
        else:
            return False
    
    # MISSION 1.1
    def selectData(self):
        # Data Selection
        st.sidebar.title("Veri Seti Secimi")
        self.selectedDataset = st.sidebar.selectbox("Lutfen bir veri seti secin:", list(self.datasets.keys()))
        
    # MISSION 1.2 and MISSION 2.4
    def showData(self):
        st.write("### Veriyi incelemeye baslamadan once sutun isimlerine bakalim:")
        st.write(f"Sutun Sayisi:{self.X.shape[1]}")
        st.write("Sutunlarin veri tiplerine ve outlier bulunup bulunmadigina kendi icinde bakalim:")
        col_names = self.df.columns.tolist()
        col_names_table = pd.DataFrame(col_names, columns=["Sutun Isimleri"])
        col_names_table["dataType"] = self.df.dtypes.values
        
        outlier_cols = []
        for col in self.df.select_dtypes(include=['float64', 'int64']).columns:
            if self.detectOutlier(col,q1=0.05,q3=0.95):
                outlier_cols.append(col)
        
        if outlier_cols:
            col_names_table["Outlier"] = col_names_table["Sutun Isimleri"].apply(lambda x: "Evet" if x in outlier_cols else "Hayir")
        else:
            col_names_table["Outlier"] = "Hayir"
        
        st.table(col_names_table)
        st.write(f"Bos Veri Sayisi: {self.df.isnull().sum().sum()}")
        st.write("### Verinin ilk 10 satirini ve son 10 satirini inceleyelim:")
        st.write("Veri Seti Ilk 10 Satiri:")
        st.write(self.df.head(10))
        st.write("Veri Seti Son 10 Satiri:")
        st.write(self.df.head(10))

    # MISSION 2.3
    def renameTarget(self):
        # Split Diagons
        self.datasets[self.selectedDataset]['diagnosis'] = self.datasets[self.selectedDataset]['diagnosis'].map({'M': 1, 'B': 0}).astype(int)
        # self.datasets[self.selectedDataset].drop("id",axis=1,inplace=True)
        self.X = self.datasets[self.selectedDataset].drop(columns=['diagnosis']) 
        self.y = self.datasets[self.selectedDataset]['diagnosis']  
        self.X.drop("id",axis=1,inplace=True)

    # MISSION 2.3    
    def showCorrelationMatrix(self):
        """
        Display the correlation matrix of the selected dataset and identify highly correlated columns.
        
        This function calculates the correlation matrix of the selected dataset and identifies columns 
        with a correlation coefficient greater than or equal to 0.90. It then visualizes the correlation 
        matrix using a heatmap. Additionally, it displays the highly correlated columns and shows another 
        correlation matrix with only those columns.
        This function must be used together with the function showSelectedCorrelation
        """
        st.write("### Korelasyon Matrisi:")
        corrMatrix = self.datasets[self.selectedDataset].drop("id",axis=True).corr()
        
        highCorrColumns = set()
        for i in range(len(corrMatrix.columns)):
            for j in range(i):
                if abs(corrMatrix.iloc[i, j]) >= 0.90:
                    highCorrColumns.add(corrMatrix.columns[i])
        if highCorrColumns:
            st.write(f"Yuksek Korelasyonlu Sutunlar: {', '.join(highCorrColumns)}")            
        corrMatrix = self.datasets[self.selectedDataset].drop("id",axis=True).corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corrMatrix, cmap='coolwarm', annot=False)
        st.pyplot()
        st.write(f"Yuksek Korelasyonlu Sutunlar Asagida Baska Bir Korelasyon Matrisi Ile Tekrar Gosterilmistir: ")
        self.SelectedCorrelation(highCorrColumns)
        
    
    # MISSION 2.4
    def showGraph(self):
        selected_feature1 = st.selectbox("Lutfen birinci ozelligi secin:", self.X.columns, index=self.X.columns.get_loc("radius_mean"))
        selected_feature2 = st.selectbox("Lutfen ikinci ozelligi secin:", self.X.columns, index=self.X.columns.get_loc("texture_mean"))
        fig, ax = plt.subplots(figsize=(10, 6))  # Figur ve eksen nesnelerini olustur
        self.df['diagnosis2'] = self.df['diagnosis'].replace({'M': 'iyi', 'B': 'kotu'})
        sns.scatterplot(data=self.df, x=selected_feature1, y=selected_feature2, hue='diagnosis2', palette={'iyi': 'green', 'kotu': 'red'}, ax=ax)
        ax.set_xlabel(selected_feature1)  # Eksen etiketlerini ayarla
        ax.set_ylabel(selected_feature2)
        ax.set_title(f"{selected_feature1} vs {selected_feature2}")
        ax.legend(title='Diagnosis')
        st.pyplot(fig)  # Olusturulan figur nesnesini st.pyplot() icine gecir
        
    # MISSION 2.5    
    def seperation(self):
        # 80%-20% data separation
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        st.write("Egitim verisi boyutu:", self.X_train.shape, self.Y_train.shape)
        st.write("Test verisi boyutu:", self.X_test.shape, self.Y_test.shape)
        
    # MISSION 3.1 3.2    
    def selectModel(self):  
        # Model Selection
        st.sidebar.title("Model Secimi")
        self.selectedModels = st.sidebar.selectbox("Lutfen bir model secin:", self.modelnames)

        if self.selectedModels == "KNN":
            self.model = KNeighborsClassifier()
            self.gridSearchKNN()
            st.sidebar.title("KNN Parametre Secimi")
            self.params['n_neighbors'] = st.sidebar.slider("Komsu Sayisi (n_neighbors)", 1, 20, self.params['n_neighbors'])
            self.params['leaf_size'] = st.sidebar.slider("Yaprak Boyutu (leaf_size)", 1, 100, self.params['leaf_size'])
            self.params['weights'] = st.sidebar.selectbox("Agirliklar (weights)", ['uniform', 'distance'], index=['uniform', 'distance'].index(self.params['weights']))
            self.params['metric'] = st.sidebar.selectbox("Mesafe Metrigi (metric)", ['euclidean', 'manhattan'], index=['euclidean', 'manhattan'].index(self.params['metric']))
            self.params['algorithm'] = st.sidebar.selectbox("Algoritma (algorithm)", ['auto', 'ball_tree', 'kd_tree', 'brute'], index=['auto', 'ball_tree', 'kd_tree', 'brute'].index(self.params['algorithm']))

        elif self.selectedModels == "Support Vector Machine":
            self.model = SVC()
            self.gridSearchSVM()
            st.sidebar.title("SVM Parametre Secimi")
            self.params['C'] = st.sidebar.select_slider("C", options=[0.1, 1, 10, 100, 1000], value=self.params['C'])
            self.params['kernel'] = st.sidebar.selectbox("Kernel", options=['linear', 'rbf', 'poly'], index=['linear', 'rbf', 'poly'].index(self.params['kernel']))
            self.params['gamma'] = st.sidebar.selectbox("Gamma", options=['scale', 'auto'], index=['scale', 'auto'].index(self.params['gamma']))
            
        elif self.selectedModels == "Naïve Bayes":
            self.model = GaussianNB() 
    # MISSION 3.3.1         
    def gridSearchKNN(self):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
            'leaf_size': [20, 30, 40],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }

        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.Y_train)

        st.write("En Iyi Parametreler:", grid_search.best_params_)
        st.write("En Iyi Skor:", grid_search.best_score_)
        self.params = grid_search.best_params_            
    # MISSION 3.3.2   
    def gridSearchSVM(self):
        # SVC icin hiperparametre aramasi yap
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
        self.grid = GridSearchCV(SVC(), param_grid, verbose=1, cv=3, n_jobs=-1)
        self.grid.fit(self.X, self.y)
        self.params = self.grid.best_params_
    # MISSION 3.4   
    def showModel(self):
        st.write("### Secilen Model:")
        st.write(self.model)
    # MISSION 4      
    def evaluateModel(self):
        y_pred = self.model.predict(self.X_test)

        # Accuracy, Precision, Recall and F1-score
        accuracy = accuracy_score(self.Y_test, y_pred)
        precision = precision_score(self.Y_test, y_pred)
        recall = recall_score(self.Y_test, y_pred)
        f1 = f1_score(self.Y_test, y_pred)

        # Confusion matrix
        cm = confusion_matrix(self.Y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
        plt.xlabel('Tahmin Edilen Degerler')
        plt.ylabel('Gercek Degerler')
        plt.title('Confusion Matrix')
        st.pyplot()

        # Sonuclari tablo seklinde yazdirma
        st.write("### Model Degerlendirmesi")
        evaluation_table = pd.DataFrame({
            'Metrics': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
            'Score': [accuracy, precision, recall, f1]
        })
        st.table(evaluation_table)
            
    def run(self):
        # MISSION 1.1
        self.selectData()
        # MISSION 2.3
        self.renameTarget()
        # MISSION 1.2 and MISSION 2.4
        self.showData()
        # MISSION 2.3 
        self.showCorrelationMatrix()
        # MISSION 2.4
        self.showGraph()
        # MISSION 2.5
        self.seperation()
        # MISSION 3.1 3.2 
        self.selectModel()
        # MISSION 3.4 
        self.showModel()
        if self.selectedModels != "Naïve Bayes":
            self.model.set_params(**self.params)
        # MISSION 3.4 
        self.model.fit(self.X, self.y)
        self.evaluateModel()

if __name__ == "__main__":
    uygulama = App()
    uygulama.run()
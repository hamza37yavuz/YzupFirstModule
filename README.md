# Yzup Mini Proje

Bu proje Milli Teknoloji Akademisi Yapay Zeka Uzmanlık Eğitimi kapsamında yapılmıştır.

## Nasıl Çalıştırırım

Projeyi kendi bilgisyarınızda çalıştırmak için aşağıdaki komutları sırayla çalıştırınız:

Terminal veya komut istemcisini açın.

Aşağıdaki komutu kullanarak projeyi klonlayın:

`git clone https://github.com/hamza37yavuz/YzupProje.git`

Terminalden app.py dosyasının bulunduğu dizine gidin:

`cd YzupProje`

Gerekli Python kütüphanelerini yüklemek için aşağıdaki komutu çalıştırın:

`pip -r requirements.txt` veya `pip3 -r requirements.txt`

Sonrasında streamlit ekranını görüntülemek için aşağıdaki komutu çalıştırın:

`python -m streamlit run .\app.py` veya `streamlit run .\app.py`

Docker kullanarak projeyi çalıştırmak isterseniz, Dockerfile dosyasının bulunduğu dizine gidin ve aşağıdaki komutları sırasıyla çalıştırın:

`docker build -t my_streamlit_app . `

`docker run -p 8501:8501 my_streamlit_app`

Komutları çalıştırdıktan sonra:

[localhost](http://localhost:8501/) linkinden projeye erişebileceksiniz.



